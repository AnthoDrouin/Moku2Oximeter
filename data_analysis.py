import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from moku.instruments import Datalogger
from sigfig import round
from typing import *
import time


class Oximeter:
	"""
	This class is used to get data from the oximeter and compute the SpO2 and BPM.
	"""

	def __init__(
			self,
			ip: str,
			patient: Optional[dict] = None,
			streaming_period: float = 10,
			filename: Optional[str] = None,
			show_data: bool = True,
			**kwargs
	):
		self.ip = ip
		self.datalogger = Datalogger(self.ip, force_connect=True)
		self.patient = patient
		self.kwargs = kwargs
		self._set_default_kwargs()
		self.filename = filename
		self.streaming_period = streaming_period
		self.show_data = show_data
		self.data = {
			"infrared" : {"ch1": [0], "time": [0]},
			"red" : {"ch1": [0], "time": [0]},
			"BPM" : None,
			"SpO2" : None,
			"all_BPMs" : None,
			"health" : [],
		}
		self.normal_hearth_rate_by_age = {
			10: (70, 190),
			20: (95, 162),
			30: (93, 157),
			40: (90, 153),
			45: (88, 149),
			50: (85, 145),
			55: (83, 140),
			60: (80, 136),
			65: (78, 132),
			70: (75, 128),
		}
		self.normal_sp02_by_pourcentage = {
			67: "DANGER : Cyanosis",
			80: "LOW OXYGEN LEVELS : Brain is affected",
			91: "Concerning blood Oxygen Levels : LOW BLOOD OXYGEN LEVELS",
			95: "Concerning blood Oxygen Levels",
			100: "normal",
		}


	def _set_default_kwargs(self):
		self.kwargs.setdefault("acquisition_mode", "Precision")
		self.kwargs.setdefault("sample_rate", 4030)
		self.kwargs.setdefault("sleep_time", 1)
		self.kwargs.setdefault("filename", None)
		self.kwargs.setdefault("plot_data", True)
		self.kwargs.setdefault("remove_first_data", 0)
		self.kwargs.setdefault("moving_average_window", 300)
		self.kwargs.setdefault("param_savgol", (400, 500))
		self.kwargs.setdefault("batch_size", 1000)


	def _set_param_streaming(self):
		"""
		Set the parameters of the streaming
		"""
		self.datalogger.set_acquisition_mode(mode=self.kwargs["acquisition_mode"])
		self.datalogger.set_samplerate(self.kwargs["sample_rate"])

	def on_acquisition_begin(self):
		"""
		Apply at the beginning of the acquisition
		"""
		self._set_param_streaming()

		self.datalogger.generate_waveform(type="DC", channel=1, dc_level=5)
		self.datalogger.generate_waveform(type="DC", channel=2, dc_level=0)


	def acquisition(self):
		"""
		Acquisition of the data
		"""
		self.on_acquisition_begin()

		self.datalogger.start_streaming(duration=self.streaming_period*2)

		print(f"Streaming started for red light {self.streaming_period}s")

		while self.data["red"]["time"][-1] < self.streaming_period:
			new_data = self.datalogger.get_stream_data()
			self.data["red"]["ch1"] += new_data["ch1"]
			self.data["red"]["time"] += new_data["time"]
			print(f"Remaining time {round(self.streaming_period - self.data['red']['time'][-1], 3)} seconds")

		self.datalogger.stop_streaming()

		self.datalogger.relinquish_ownership()
		time.sleep(3)
		self.datalogger = Datalogger(self.ip, force_connect=True)
		self._set_param_streaming()


		self.datalogger.generate_waveform(type="DC", channel=1, dc_level=0)
		self.datalogger.generate_waveform(type="DC", channel=2, dc_level=5)

		self.datalogger.start_streaming(duration=self.streaming_period)


		print(f"Streaming started for infrared light {self.streaming_period}s")
		while self.data["infrared"]["time"][-1] < self.streaming_period:
			new_data = self.datalogger.get_stream_data()
			self.data["infrared"]["ch1"] += new_data["ch1"]
			self.data["infrared"]["time"] += new_data["time"]
			print(f"Remaining time {round(self.streaming_period - self.data['infrared']['time'][-1], 3)} seconds")

		self.datalogger.stop_streaming()

		plt.plot(self.data["infrared"]["time"], self.data["infrared"]["ch1"])
		plt.plot(self.data["red"]["time"], self.data["red"]["ch1"])
		plt.show()

		self.on_acquisition_end()


	def on_acquisition_end(self):
		"""
		Apply at the end of the acquisition
		"""
		self.data["infrared"]["ch1"] = np.array(self.data["infrared"]["ch1"][1:])
		self.data["infrared"]["time"] = np.array(self.data["infrared"]["time"][1:])
		self.data["red"]["ch1"] = np.array(self.data["red"]["ch1"][1:])
		self.data["red"]["time"] = np.array(self.data["red"]["time"][1:])

		self.data["infrared"]["ch1"] = Oximeter.apply_savgol_filter(self.data["infrared"]["ch1"], 51, 3)
		self.data["red"]["ch1"] = Oximeter.apply_savgol_filter(self.data["red"]["ch1"], 51, 3)

		bpm_red = Oximeter.compute_bpm(
			self.data["red"],
			remove_first_data=self.kwargs.get("remove_first_data", 0),
			param_savgol=self.kwargs.get("param_savgol", (200, 500)),
			moving_average_window=self.kwargs.get("moving_average_window", 300),
		)
		bpm_infrared = Oximeter.compute_bpm(
			self.data["infrared"],
			remove_first_data=self.kwargs.get("remove_first_data", 0),
			param_savgol=self.kwargs.get("param_savgol", (200, 500)),
			moving_average_window=self.kwargs.get("moving_average_window", 300),
		)
		self.data["BPM"] = (bpm_red[0] + bpm_infrared[0])/ 2

		self.data["all_BPMs"] = [bpm_red[0], bpm_infrared[0]]

		self.data["SpO2"] = Oximeter.compute_spO2(
			data=self.data,
			param_savgol=(int(-27.5174*self.data["BPM"] + 4254.2561),None),
			batch_size=int(149.98488*self.data["BPM"] - 8183.47294)
		)
		# param_savgol and batch_size were determine empirically
		self.check_health()

		if self.filename is not None:
			np.save(self.filename + ".npy", self.data)



	def check_health(self) -> None:
		assert isinstance(self.patient, dict), "the patient informations must be a dict"
		assert self.patient.get("age") is not None, "The age of the patient must be provided to determine it's health"

		for age in self.normal_hearth_rate_by_age.keys():
			if self.patient["age"] < age:
				if self.data["BPM"] < self.normal_hearth_rate_by_age[age][0]:
					self.data["health"].append(f"bpm under the average {self.normal_hearth_rate_by_age[age][0]} - {self.normal_hearth_rate_by_age[age][1]}")
				elif self.data["BPM"] > self.normal_hearth_rate_by_age[age][1]:
					self.data["health"].append(f"bpm above the average {self.normal_hearth_rate_by_age[age][0]} - {self.normal_hearth_rate_by_age[age][1]}")
				else:
					self.data["health"].append(f"Your BPM is as expected for you age. Your BPM is in the following range {self.normal_hearth_rate_by_age[age][0]} - {self.normal_hearth_rate_by_age[age][1]}")
				break
			else:
				continue
		if 70 < self.patient["age"]:
			self.data["health"].append(f"No data is available for your age. Sorry :(")

		for normal_concentration in self.normal_sp02_by_pourcentage.keys():
			if self.data["SpO2"][0] < normal_concentration:
				self.data["health"].append(f"{self.normal_sp02_by_pourcentage[normal_concentration]}")
				break


	@staticmethod
	def extract_data(path, to_dict: bool = False) -> Union[dict, Tuple[np.ndarray, np.ndarray]]:
		"""
		Extract the data from the file
		:param path: path to the data
		:param to_dict: if True, return a dict with the time and the tension data
		:return: time and tension data
		"""
		tension = []
		time = []
		try:
			data = np.load(path)
			for t, v in data:
				tension.append(v)
				time.append(t)

		except:
			data = np.load(path, allow_pickle=True).item()

			for t in data["time"]:
				time.append(t)

			for v in data["ch1"]:
				tension.append(v)
		if not to_dict:
			return np.array(time), np.array(tension)
		else:
			return {"time": np.array(time), "ch1": np.array(tension)}

	@staticmethod
	def compute_spO2(
			data: Optional[dict] = None,
			path: Optional[str] = None,
			remove_first_data: Optional[int] = None,
			param_savgol: Tuple[int] = (3025, 500),
			batch_size: int = 2000,
	) -> Tuple[float, float]:
		"""
		Compute the SpO2
		:param data: data to compute the SpO2
		:param path: path to the data
		:param remove_first_data: remove the first data
		:param param_savgol: parameters for the savgol filter
		:param batch_size: size of the batch
		:return: SpO2 and the confidence
		"""
		time_red, tension_red = Oximeter.prepare_data(data["red"], path)
		time_ir, tension_ir = Oximeter.prepare_data(data["infrared"], path)
		if remove_first_data is not None:
			time_red, tension_red = time_red[remove_first_data:], tension_red[remove_first_data:]
			time_ir, tension_ir = time_ir[remove_first_data:], tension_ir[remove_first_data:]

		time_red_f1, tension_red_f1 = time_red, Oximeter.apply_savgol_filter(tension_red, param_savgol[0])
		time_ir_f1, tension_ir_f1 = time_ir, Oximeter.apply_savgol_filter(tension_ir, param_savgol[0])

		time_red_f1 = time_red_f1[1::]
		tension_red_f1 = tension_red_f1[1::]

		len_data_red = len(time_red_f1)
		len_data_ir = len(time_ir_f1)
		if len_data_ir > len_data_red:
			time_ir_f1 = time_ir_f1[:len_data_red]
			tension_ir_f1 = tension_ir_f1[:len_data_red]
		elif len_data_red > len_data_ir:
			time_red_f1 = time_red_f1[:len_data_ir]
			tension_red_f1 = tension_red_f1[:len_data_ir]


		len_data = len(time_red_f1)

		# seperate data in batch_size
		batches_red = [tension_red_f1[i:i + batch_size] for i in range(0, len_data, batch_size)]
		batches_ir = [tension_ir_f1[i:i + batch_size] for i in range(0, len_data, batch_size)]

		spo2_predicted = []

		for idx, batch in enumerate(zip(batches_red, batches_ir)):
			batch_red, batch_ir = batch
			ac_red_div_dc_red = (np.max(batch_red) - np.min(batch_red))/np.min(batch_red)
			ac_ir_div_dc_ir = (np.max(batch_ir) - np.min(batch_ir))/np.min(batch_ir)
			spo2_predicted.append(110 - 25 * ac_red_div_dc_red/ac_ir_div_dc_ir)
			#spo2_predicted.append(110 - 25 * np.mean(batch_red)/np.mean(batch_ir))
			#spo2_predicted.append(110 - 25 * 630/940 * np.log10(np.mean(batch_red)) / np.log10(np.mean(batch_ir)))
			#R = ac_red_div_dc_red/ac_ir_div_dc_ir
			#spo2_predicted.append((1.5958422 * R**2) -34.6596622*R + 112.6899759)

		return np.mean(spo2_predicted), np.std(spo2_predicted)

	@staticmethod
	def compute_bpm(
			data: Optional[dict] = None,
			path: Optional[str] = None,
			remove_first_data: Optional[int] = None,
			param_savgol: Tuple[int] = (200, 500),
			moving_average_window: int = 300,
	) -> Tuple[np.ndarray, dict]:
		"""
		Compute the BPM from the data
		:param path: path to the data
		:return: time and tension data
		"""
		time, tension = Oximeter.prepare_data(data, path)

		if remove_first_data is not None:
			time = time[remove_first_data:]
			tension = tension[remove_first_data:]

		time_f1, tension_f1 = time, Oximeter.apply_savgol_filter(tension, param_savgol[0])

		tension_diff = Oximeter.differentiate(tension_f1)

		time_f2, tension_diff_f2 = time_f1, Oximeter.apply_savgol_filter(tension_diff, param_savgol[1])

		tension_diff_f2_mean = Oximeter.moving_average(tension_diff_f2, moving_average_window)

		bpm, idx_min = Oximeter._compute_bpm_from_signal(time_f2, tension_diff_f2_mean, moving_average_window)

		extra_params = {
			"index_minimum_time": idx_min[0],
			"index_minimum": idx_min[1],
			"time": time,
			"tension_f1": tension_f1
		}

		return bpm, extra_params

	@staticmethod
	def prepare_data(data: Optional[dict], path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Prepare the data
		:param data: data to prepare
		:param path: path to the data
		:return: time and tension data
		"""
		if path is None and data is None:
			raise ValueError("You must specify a path or give a dataset")
		if path is not None:
			time, tension = Oximeter.extract_data(path)
		else:
			assert data.get("time") is not None and data.get("ch1") is not None, "The data must have a 'time' and a 'ch1' key"
			time, tension = data["time"], data["ch1"]

		return time, tension


	@staticmethod
	def apply_savgol_filter(data_to_filter, window_lenghts: int = 500, polyorder: int = 3) -> np.ndarray:
		"""
		Apply a Savitzky-Golay filter to the data
		:param data_to_filter: data to filter
		:param window_lenghts: The length of the filter window (i.e., the number of coefficients).
		:param polyorder: The order of the polynomial used to fit the samples.
		:return: filtered data
		"""
		return scipy.signal.savgol_filter(data_to_filter, window_lenghts, polyorder)

	@staticmethod
	def differentiate(data_to_differentiate: np.ndarray) -> np.ndarray:
		"""
		Differentiate the data
		:param data_to_differentiate: data to differentiate
		:return: differentiated data
		"""
		return np.diff(data_to_differentiate)

	@staticmethod
	def moving_average(data_to_average: np.ndarray, window:int = 300) -> np.ndarray:
		"""
		Apply a moving average to the data
		"""
		return np.convolve(data_to_average, np.ones(window), 'valid') / window

	@staticmethod
	def _compute_bpm_from_signal(time: np.ndarray, tension: np.ndarray, offset: int) -> float:
		"""
		Compute the BPM from a signal that is already filtered
		:param time: Array of time
		:param tension: Array of tension
		:param offset: Offset to add to the index of the minimum -> Needed if moving average is used
		:return: BPM
		"""
		mean = np.mean(tension)
		std = np.std(tension)
		all_minimums = scipy.signal.argrelextrema(tension, np.less)[0]
		minimums_under_threshold = np.where(tension[all_minimums] < mean - 2 * std)

		index_minimum = all_minimums[minimums_under_threshold]
		index_minimum_time = np.array(index_minimum) + offset

		num_hearthbeat = len(index_minimum_time)
		dt_heathnbeat = time[index_minimum_time[-1]] - time[index_minimum_time[0]]
		bpm = num_hearthbeat / dt_heathnbeat * 60
		return bpm, (index_minimum_time, index_minimum)

	@staticmethod
	def plot_data(time: np.ndarray, tension: np.ndarray, type_graph: str = "final", show: bool = True, **kwargs) -> plt.Figure:
		"""
		Plot the data
		:param time:
		:param tension:
		:param type_graph: "final", "raw", "detection"
		:param kwargs:
		:return:
		"""
		plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")
		fig = plt.figure(figsize=(16, 5))
		if type_graph == "final":
			plt.plot(time, tension, "-")
			if kwargs.get("index_minimum_time") is not None:
				plt.scatter(time[kwargs.get("index_minimum_time")], tension[kwargs.get("index_minimum_time")])
			plt.xlabel("Temps [s]", fontsize=22)
			plt.ylabel("Tension [V]", fontsize=22)
			if kwargs.get("title") is not None:
				plt.title(kwargs.get("title"), fontsize=22)
		elif type_graph == "raw":
			raise NotImplementedError
		elif type_graph == "detection":
			raise NotImplementedError
		else:
			raise ValueError("type_graph must be 'final', 'raw' or 'detection'")
		if show:
			plt.show()


#if __name__ == '__main__':
	#data_dict = Oximeter.extract_data("data_antoine.npy", to_dict=True)
	#data_dict = Oximeter.extract_data("data_antho_real_good.npy", to_dict=True)
	# bpm, extra = Oximeter.compute_bpm(data=data_dict, remove_first_data=100)
	# Oximeter.plot_data(
	# 	extra["time"],
	# 	extra["tension_f1"],
	# 	type_graph="final",
	# 	index_minimum_time=extra["index_minimum_time"],
	# 	index_minimum=extra["index_minimum"],
	# 	bpm=bpm
	# )
