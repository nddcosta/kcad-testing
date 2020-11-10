import sys, os
sys.path.append("../")

from config.config import KCADArgParser
import pandas as pd
from core.kcad_raw_data_manager import *
from core.kcad_learning import *
from core.kcad_line_segment_parser import *
from config.config import *
from pathlib import Path
from enum import Enum
import numpy as np
from multiprocessing import Pool, Process
from itertools import product
import time
import random
import itertools
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, auc
from utils import *
from os.path import basename, normpath
	
	

class DanglingFeaturesInDataset(Exception):
	pass



def multi_arg_wrapper(args):
	return TestManager.worker_compile_result_rows(*args)



def make_directories(dir_list):
	for directory in dir_list:
		if not directory.exists():
			os.mkdir(str(directory))



#an adversary will have always have the same number of lines as the original segment file and will have modifications on line_nums
class Adversary:
	def __init__(self, advs_name, label, line_nums, segment_file_path, adv_type, percent_size, multiplier, modification, segments_frame):
		self.name = advs_name
		self.label = label
		self.line_nums = line_nums
		self.segment_file_path = segment_file_path
		self.adv_type = adv_type
		self.multiplier = multiplier
		self.segments_frame = segments_frame
		self.modification = modification
		self.percent_size = percent_size

	def generate_csv_file(self):
		self.segments_frame.to_csv(str(self.segment_file_path))

	def read_csv_file(self):
		self.segments_frame = pd.read_csv(str(self.segment_file_path))
		if "Unnamed: 0" in self.segments_frame.columns:
			self.segments_frame.drop(["Unnamed: 0"], axis=1, inplace=True)

	def __str__(self):
		return self.name



#responsible for creating/getting adversaries
class AdversaryManager:
	def __init__(self, config, error_thresholds, mod_lines):
		self.adversaries = []
		self.mod_lines = mod_lines
		self.bin_labels = [label for label in config.cal_labels_list if LabelType(label[0]) == LabelType.BINARY]
		self.cont_labels = [label for label in config.cal_labels_list if LabelType(label[0]) == LabelType.CONTINUOUS]
		self.mod_multipliers = [.25, .5, .75, 1.0, 1.25, 1.5]

		#creates modifications for continuous labels that are mod_multiplied on the error thresholds
		self.label_mod_degrees = {label:[(value*multiplier, str(int(multiplier*100))) for multiplier in self.mod_multipliers] for label, value in error_thresholds.items() if label in self.cont_labels}

		self.advs_basedir = Path(config.data_path) / "adversary"
		self.advs_cont_basedir = self.advs_basedir / "continuous"
		self.advs_bin_basedir = self.advs_basedir / "binary"

		make_directories([self.advs_basedir, self.advs_cont_basedir, self.advs_bin_basedir])

		line_segment_manager = LineSegmentManager(config.data_path)
		line_segment_manager.parse(Path(config.data_path)/'data'/'Timing.csv')

		self.segments_frame = line_segment_manager.segments_frame


	def modifiy_lines(self, label_type, item, percent_size):
		adv_kwargs = {"advs_name":None, "segment_file_path":None,"label": None,
				"line_nums":item, "adv_type": None, "percent_size":percent_size,
				"multiplier":None, "modification":None, "segments_frame":None}

		if label_type == LabelType.CONTINUOUS:
			for label in self.cont_labels:
				for modification, multiplier in self.label_mod_degrees[label]:
					adv_kwargs["advs_name"] = label + "_Mod_size_" + str(modification) + "_Mod_percent_" + str(multiplier)+ "_Num_mods_" + percent_size + "_percent"
					adv_kwargs["segment_file_path"] = self.advs_cont_basedir / (adv_kwargs["advs_name"] + ".csv")
					adv_kwargs["modification"] = modification
					adv_kwargs["multiplier"] = multiplier
					adv_kwargs["label"] = label
					adv_kwargs["adv_type"] = LabelType.CONTINUOUS

					if not adv_kwargs["segment_file_path"].exists():
						temp_dataframe = self.segments_frame.copy()
						for linenum in item:
							label_value = self.segments_frame.at[linenum, label]
							modified_value = label_value + modification
							temp_dataframe.at[linenum, label] = modified_value 
						adv_kwargs["segments_frame"] = temp_dataframe
						adversary = Adversary(**adv_kwargs)
						adversary.generate_csv_file()
					else:
						adversary = Adversary(**adv_kwargs)
						adversary.read_csv_file()
					self.adversaries.append(adversary)

		elif label_type == LabelType.BINARY:
			for label in self.bin_labels:
				adv_kwargs["advs_name"] = label + "_Num_mods_" + percent_size
				adv_kwargs["segment_file_path"] = self.advs_bin_basedir / (adv_kwargs["advs_name"] + ".csv")
				adv_kwargs["label"] = label
				adv_kwargs["adv_type"] = LabelType.BINARY
			
				if not adv_kwargs["segment_file_path"].exists():
					temp_dataframe = self.segments_frame.copy()
					for linenum in item:
						label_value = self.segments_frame.at[linenum, label]
						temp_dataframe.at[linenum, label] = int(not label_value)
					adv_kwargs["segments_frame"] = temp_dataframe
					adversary = Adversary(**adv_kwargs)
					adversary.generate_csv_file()
				else:
					adversary = Adversary(**adv_kwargs)
					adversary.read_csv_file()
				self.adversaries.append(adversary)


	def get_generated_adversaries(self):
		for item, percent_size  in self.mod_lines:
			self.modifiy_lines(LabelType.BINARY, item, percent_size)
			self.modifiy_lines(LabelType.CONTINUOUS, item, percent_size)



class Test:
	def __init__(self, original_labels, test_labels, models, feature_dict, error_thresholds, testing_label):
		self.test_labels = test_labels
		self.testing_label = testing_label
		self.models = models
		self.feature_dict = feature_dict
		self.error_thresholds = error_thresholds
		self.original_labels = original_labels
		self.detection_info = {}
		self.error_info = {}
		self.check_modification()


	#in real test, modification is not assumed, in generated test, check for modification on testing label only
	def check_modification(self):
		#allows for some very slight differences due to occasional loss of least signifcant decimal place in copied segment file
		def values_close_enough(original, modified, small_percentage=.00001):
			return ((original <= (modified + abs(small_percentage * modified))) and (original >= (modified - abs(small_percentage * modified))))

		if self.testing_label is not None and self.testing_label in self.models:
			if values_close_enough(self.original_labels[self.testing_label], self.test_labels[self.testing_label]):
				self.modified = 0
			else:
				self.modified = 1
		else:
			self.modified = np.nan


	#in real test, detection determined if any label is detected, in generated test, detection is determined only if testing label is detected
	def run_detection(self):
		for label, (model, _) in self.models.items():
			feature_array = self.feature_dict[label].reshape(1,-1)
			predicted_label_value = Evaluator.get_prediction(model, label, feature_array)
			self.error_info[label] = abs(predicted_label_value - self.test_labels[label])

			if(self.error_info[label] > self.error_thresholds[label]):
				self.detection_info[label] = 1
			else:
				self.detection_info[label] = 0

		if self.testing_label is not None:
			self.detected = self.detection_info[self.testing_label]
		else:
			if any(detection for label, detection in self.detection_info.items()):
				self.detected = 1
			else:
				self.detected = 0



#gathers metrics on each tested adversary and puts results into one csv, only for generated tests
class Results:
	def __init__(self, adv_manager, error_thresholds, feature_results_path, segment_results_path, file_results_path, test_results_path):
		self.adv_manager = adv_manager

		self.file_tests = pd.read_csv(file_results_path)
		self.segment_tests = pd.read_csv(segment_results_path)
		self.feature_tests = pd.read_csv(feature_results_path)
		self.error_thresholds = error_thresholds
		self.test_results_path = test_results_path
		self.figures_path = self.test_results_path / "figures"

		self.overall_results_path = self.test_results_path  / "overall_test_results.csv"

		make_directories([self.figures_path])


		self.results_frame = pd.DataFrame(columns=["Granularity", "Test", "Accuracy", "TPR", "FPR", "F1_score", "AUC_score"])


	def evaluate_all(self):
		test_index = 0
		roc_groups = {}
		for tests in [("File", self.file_tests), ("Segment", self.segment_tests), ("Feature",self.feature_tests)]:
			for adv in self.adv_manager.adversaries:

				testing_set_y = self.get_testing_set_Y(tests[1], adv.name).values.tolist()
				predicted_set_y = self.get_predicted_set_Y(tests[1], adv.name).values.tolist()

				#groups ROC curves by binary labels by label and continous labels by label + number of modifications
				if tests[0] != "File":
					if adv.adv_type == LabelType.BINARY:
						graph_name = adv.label + "_" + tests[0]
					elif adv.adv_type == LabelType.CONTINUOUS:
						graph_name = adv.label + "_" + tests[0] + "_Num_mod_percentage_" + adv.percent_size

					if graph_name not in roc_groups:
						roc_groups[graph_name] = []

					roc_groups[graph_name].append((adv.name, testing_set_y, predicted_set_y))	
				
				if len(testing_set_y) == 0 or len(predicted_set_y) == 0: continue #no tests in the subtest

				tp, fp, tn, fn = self.perf_measure(testing_set_y, predicted_set_y)

				result_row = {"Granularity": tests[0],
								"Test": adv.name,
								"Accuracy": accuracy_score(testing_set_y, predicted_set_y),
								"F1_score": f1_score(testing_set_y, predicted_set_y, average='binary'),
								"AUC_score": auc(*roc_curve(testing_set_y, predicted_set_y)[0:2]),
								"TPR": (tp / (tp + fn)) if (tp + fn) > 0 else np.nan,
								"FPR": (fp / (fp + tn)) if (fp + tn) > 0 else np.nan}

				self.results_frame.loc[test_index] = pd.Series(result_row)
				test_index += 1
		
		for graph_name, adv_list in roc_groups.items():
			self.get_roc_curve(graph_name, adv_list) 

		self.results_frame.to_csv(str(self.overall_results_path), index=False)


	def perf_measure(self, testing_set_y, predicted_set_y):
		TP = FP = TN = FN = 0
		
		for i in range(len(predicted_set_y)):
			if testing_set_y[i]==predicted_set_y[i]==1:
				TP += 1
			if predicted_set_y[i]==1 and testing_set_y[i]!=predicted_set_y[i]:
				FP += 1
			if testing_set_y[i]==predicted_set_y[i]==0:
				TN += 1
			if predicted_set_y[i]==0 and testing_set_y[i]!=predicted_set_y[i]:
				FN += 1

		return(TP, FP, TN, FN)


	def get_testing_set_Y(self, tests, sub_test):
		return tests['{}_modified'.format(sub_test)]


	def get_predicted_set_Y(self, tests, sub_test):
		return tests['{}_detected'.format(sub_test)]


	def get_roc_curve(self, group_name, adv_tests):
		plt.figure(figsize=(12,12))
		plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
		for adv_name, testing_set_y, predicted_set_y in adv_tests:
			fpr, tpr, thresholds = roc_curve(testing_set_y, predicted_set_y)
			roc_auc = auc(fpr, tpr)
			plt.plot(fpr,tpr, label='{} (area={})'.format(adv_name, roc_auc))
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.0])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.suptitle("Receiver Operating Characteristic for " + group_name)
		plt.title("Num_mods = percentage_of_all_segments" + "_Error_thresholds = " + str(self.error_thresholds))
		plt.legend(loc="lower right")
		plt.savefig(fname=str(Path(self.test_results_path /'figures'/'ROC_scores_{}.png'.format(group_name))), bbox_inches='tight', dpi=300)



#base test manager class responsible for creating and running tests as well as capturing results
class TestManager:
	def __init__(self, config, error_thresholds, models_dict, mod_lines=None):
		self.config = config
		self.label_manager = LabelManager(config, models_dict)
		self.mod_lines = mod_lines
		self.error_thresholds = error_thresholds
		self.working_path = Path(config.data_path)
		self.testing_path = self.working_path / "csv" / "kcad_testing"

		self.result_columns_dict = {}
		self.result_columns_dict["detailed_features"] = ["name", "segment_idx", "feature_idx", "original_labels", "models" ,"test_labels", "modified", "detected",
														"modification_size"] + ["{}_detected".format(label) for label in self.config.cal_labels_list] + \
														["{}_error".format(label) for label in self.config.cal_labels_list]

		self.result_columns_dict["features"] = ["modified", "detected"]
		self.result_columns_dict["segments"] = ["modified", "detected", "detection_rate"]
		self.result_columns_dict["file"]     = ["modified", "detected", "segments_detected", "total_segments", "features_detected", "total_features"] +\
												["total_{}_detected".format(label) for label in self.config.cal_labels_list]

		make_directories([self.testing_path])
		self.input_frame = pd.read_csv(str(config.input_path))

	
	def create_file_paths(self, test_name):
		self.test_name = "{}__thresholds_Vx-{}_Vy-{}_File-{}".format(self.config.testing_filename,
																	str(self.error_thresholds["Vx"]),
																	str(self.error_thresholds["Vy"]),
																	str(self.error_thresholds["File"]))
		self.test_results_path = self.test_results_dir / self.test_name

		self.detailed_feature_detection_path = self.test_results_path / ("detailed_feature_results.csv")
		self.feature_detection_path = self.test_results_path / ("feature_results.csv")
		self.segment_detection_path = self.test_results_path / ("segment_results.csv")
		self.file_detection_path = self.test_results_path / ("file_results.csv")
		make_directories([self.test_results_dir, self.test_results_path])


	def create_csv_files(self, detailed_feature_detections, feature_detections, segment_detections, file_detections):
		detailed_feature_detection_tests = pd.concat(detailed_feature_detections)
		detailed_feature_detection_tests.to_csv(str(self.detailed_feature_detection_path))

		feature_detection_tests = pd.concat(feature_detections, axis=1)
		feature_detection_tests.to_csv(str(self.feature_detection_path))

		segment_detection_tests = pd.concat(segment_detections, axis=1)
		segment_detection_tests.to_csv(str(self.segment_detection_path))

		file_detection_tests = pd.concat(file_detections, axis=1)
		file_detection_tests.to_csv(str(self.file_detection_path), index=False)


	#ran for each individual adversary, records results on a per file, per segment and per feature basis
	@staticmethod
	def worker_compile_result_rows(label_manager, error_thresholds, input_frame, segments_frame, result_columns_dict, test_itself=None, adversary=None):
		test_index = 0
		total_segment_detections = 0
		total_feature_detections = 0
		label_detections_dict = dict.fromkeys(label_manager.labels, 0)
		file_info = TestManager._get_file_test_info(segments_frame, adversary)

		detailed_feature_results_frame = pd.DataFrame(columns=result_columns_dict["detailed_features"])
		feature_results_frame = pd.DataFrame(columns=[file_info["name"]+"_"+column for column in result_columns_dict["features"]])
		segment_results_frame = pd.DataFrame(columns=[file_info["name"]+"_"+column for column in result_columns_dict["segments"]])
		file_results_frame = pd.DataFrame(columns=[file_info["name"]+"_"+column for column in result_columns_dict["file"]])

		file_test_results = dict.fromkeys(result_columns_dict["file"], np.nan)

		print("generating tests for adversary:", file_info["name"])
		for segment_index, row in file_info["test_frame"].iterrows():
			segment_test_results = dict.fromkeys(result_columns_dict["segments"], np.nan)

			test_labels = TestManager._get_labels(row)

			#gets original labels from segment.csv for the segment index, used to check for modification, sets to None for real test
			original_labels = TestManager._get_labels(segments_frame.iloc[segment_index]) if adversary is not None else None

			segment_frame = input_frame[input_frame["segment_idx"] == segment_index]
			if segment_frame.size == 0: continue

			segment_feature_detections = 0
			for index, _ in segment_frame.iterrows():
				feature_test_results = dict.fromkeys(result_columns_dict["detailed_features"], np.nan)

				test_kwargs = {"models":label_manager.models_dict, "feature_dict":label_manager.get_feature_rows(index),
							"original_labels":original_labels, "test_labels":test_labels, "error_thresholds": error_thresholds,
							"testing_label":file_info["testing_label"]}

				test = Test(**test_kwargs)
				test.run_detection()

				feature_test_results["name"]              = file_info["name"]
				feature_test_results["modified"] 		  = test.modified
				feature_test_results["detected"] 		  = test.detected
				feature_test_results["segment_idx"]       = segment_index
				feature_test_results["feature_idx"]       = index
				feature_test_results["original_labels"]   = original_labels
				feature_test_results["models"]			  = label_manager.model_names
				feature_test_results["test_labels"]       = test_labels

				for label in label_manager.labels:
					feature_test_results[("{}_detected").format(label)] = test.detection_info[label]
					feature_test_results[("{}_error").format(label)] = test.error_info[label]
					label_detections_dict[label] += test.detection_info[label]

				detailed_result_row = {col:feature_test_results[col] for col in result_columns_dict["detailed_features"]}
				detailed_feature_results_frame.loc[test_index] = pd.Series(detailed_result_row)

				feature_results_row = TestManager._get_result_row(file_info["name"], feature_test_results, result_columns_dict["features"])
				feature_results_frame.loc[test_index] = pd.Series(feature_results_row)

				segment_feature_detections += feature_test_results["detected"]
				total_feature_detections += feature_test_results["detected"]
				test_index += 1

			#if there are more detections than not for all the feature rows in a given segment assume segment overall is detected
			segment_test_results["segment_detection_rate"] = float(segment_feature_detections) / len(segment_frame.index)
			segment_test_results["modified"] = (1 if segment_index in file_info["modified_lines"] else 0) if file_info["modified_lines"] is not None else np.nan
			segment_test_results["detected"] = 1 if segment_test_results["segment_detection_rate"] >= .5 else 0

			segment_result_row = TestManager._get_result_row(file_info["name"], segment_test_results, result_columns_dict["segments"])
			segment_results_frame.loc[segment_index] = pd.Series(segment_result_row)

			total_segment_detections += segment_test_results["detected"]

		file_test_results["total_segments"]     = len(segment_frame.index)
		file_test_results["total_features"]     = test_index
		file_test_results["segments_detected"]  = total_segment_detections
		file_test_results["feautres_detected"]  = total_feature_detections
		file_test_results["total_Vx_detected"]  = label_detections_dict.get("Vx", np.nan)
		file_test_results["total_Vy_detected"]  = label_detections_dict.get("Vy", np.nan)
		file_test_results["total_Ax_detected"]  = label_detections_dict.get("Ax", np.nan)
		file_test_results["total_Ay_detected"]  = label_detections_dict.get("Ay", np.nan)
		file_test_results["total_Axy_detected"] = label_detections_dict.get("Axy", np.nan)
		file_test_results["total_Vx_detected"]  = label_detections_dict.get("Vx", np.nan)
		file_test_results["modified"]           = (1 if (len(file_info["modified_lines"])) > 0 else 0) if file_info["modified_lines"] is not None else ("file_modified" if not test_itself else "file_unmodified")
		file_test_results["detected"]           = 1 if (total_segment_detections - error_thresholds["File"]) > 0 else 0

		file_result_row = TestManager._get_result_row(file_info["name"], file_test_results, result_columns_dict["file"])
		file_results_frame.loc[0] = pd.Series(file_result_row)

		return detailed_feature_results_frame, feature_results_frame, segment_results_frame, file_results_frame


	@staticmethod
	def _get_file_test_info(segments_frame, adversary=None):
		file_test_info = {}
		if adversary is not None:
			file_test_info["name"] = adversary.name
			file_test_info["modified_lines"] = adversary.line_nums
			file_test_info["test_frame"] = adversary.segments_frame
			file_test_info["testing_label"] = adversary.label
		else:
			file_test_info["name"] = "Real_Test"
			file_test_info["modified_lines"] = None
			file_test_info["test_frame"] = segments_frame
			file_test_info["testing_label"] = None
		return file_test_info


	@staticmethod 
	def _get_result_row(name, test_values, result_columns):
		return {name+"_"+col:test_values[col] for col in result_columns}


	@staticmethod
	def _get_labels(row):
		return {"Vx": row["Vx"], "Vy": row["Vy"], "Vz": row["Vz"],
				"Ax": row["Ax"], "Ay": row["Ay"], "Az": row["Az"],
				"Axy": row["Axy"], "Axz": row["Axz"], "Ayz": row["Ayz"]}


	def run(self, func_args):
		frames = [multi_arg_wrapper(func_arg) for func_arg in func_args]
		self.create_csv_files(*zip(*frames))
			


#uses the -ipt input.csv to test against the -adv segment.csv, using models trained with -dpt 
class RealTestManager(TestManager):
	def __init__(self, config, error_thresholds, models_dict):
		super().__init__(config, error_thresholds, models_dict)
		self.test_results_dir = self.testing_path / "real_tests"
		self.create_file_paths(config.testing_filename)
		line_segment_manager = LineSegmentManager(config.adv_data_path)
		line_segment_manager.parse(Path(config.adv_data_path)/'data'/'Timing.csv')
		self.segments_frame = line_segment_manager.segments_frame


	def run_tests(self):
		func_args = [(self.label_manager, self.error_thresholds, self.input_frame, self.segments_frame, self.result_columns_dict, self.config.data_path==self.config.adv_data_path)]
		self.run(func_args)


#runs tests and evaluates results for each generated adversary in the adversary path
class GeneratedTestManager(TestManager):
	def __init__(self, config, error_thresholds, models_dict, mod_lines):
		super().__init__(config, error_thresholds, models_dict, mod_lines)
		self.test_results_dir = self.testing_path / "generated_tests"
		self.create_file_paths(config.testing_filename)
		line_segment_manager = LineSegmentManager(config.data_path)
		line_segment_manager.parse(Path(config.data_path)/'data'/'Timing.csv')
		self.segments_frame = line_segment_manager.segments_frame

		
	def get_testing_adversaries(self):
		self.adversary_manager = AdversaryManager(self.config, self.error_thresholds, self.mod_lines)
		self.adversary_manager.get_generated_adversaries()


	def evaluate_results(self):
		self.results = Results(self.adversary_manager, self.error_thresholds, self.feature_detection_path, self.segment_detection_path, self.file_detection_path, self.test_results_path)
		self.results.evaluate_all()


	def run_tests(self):
		func_args = [(self.label_manager, self.error_thresholds, self.input_frame, self.segments_frame, self.result_columns_dict, None, adversary) for adversary in self.adversary_manager.adversaries]
		self.run(func_args)


#responsible for managing labels, their associated models, and their feature frames
class LabelManager:
	def __init__(self, config, models_dict):
		self.config = config
		#if -dpt path != -ipt path create new feature files filtering the -ipt input with -dpt columns
		if self.config.data_set_path != self.config.data_path:
			self.csv_path = Path(self.config.data_path) / 'csv' / (self.config.testing_filename + '_inputs')
			if not self.csv_path.exists():
				os.mkdir(str(self.csv_path))

			ipt_input = pd.read_csv(str(self.config.input_path), index_col=0)
			for label, (_, input_manager) in models_dict.items():
				label_csv_path = Path(self.csv_path / ('%s.csv'%label))
				if not label_csv_path.exists():
					dpt_X = input_manager.input_X
					dpt_X.drop([label], axis=1)
					dangling_columns = set(dpt_X.columns) - set(ipt_input.columns)
					if len(dangling_columns) > 0:
						raise DanglingColumnsInDataset("Dangling column(s) present in dataset: {}, delete column(s) from {}.csv in -dpt path".format(dangling_columns, label))
					ipt_input[dpt_X.columns].to_csv(str(label_csv_path))
		else:
			self.csv_path = Path(self.config.data_path) / 'csv'
		self.models_dict = models_dict
		self.model_names = {label:model[0][0] for label, model in self.models_dict.items()}
		self.labels = config.cal_labels_list
		self.feature_dict = self.get_feature_dict()

	
	#each label mapped to its corresponding feature_inputs, transformed 
	def get_feature_dict(self):
		feature_dict = {}
		for label in self.labels:
			label_frame = pd.read_csv(str(Path(self.csv_path / ('%s.csv'%label))), index_col=0)
			if "Unnamed: 0.1" in label_frame.columns:
				label_frame = label_frame.drop(["Unnamed: 0.1"], axis=1)
			input_X = label_frame.drop([label], axis=1)
			input_manager = self.models_dict[label][1]
			feature_dict[label] = np.asarray(input_manager.scaler.transform(input_X))
		return feature_dict


	def get_feature_rows(self, index):
		feature_rows = {}
		for label, scaled_input_X in self.feature_dict.items():
			feature_rows[label] = scaled_input_X[index]
		return feature_rows


