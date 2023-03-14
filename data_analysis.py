import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from moku.instruments import Datalogger
from typing import *


class Oximeter:

	def __init__(
			self,
			datalogger: "Datalogger",
			patient: dict,
			**kwargs
	):
		self.datalogger = datalogger
		self.patient = patient
		self.kwargs = kwargs
		self.data = {
			"infrared" : None,
			"red" : None,
			"BPM" : None,
			"SpO2" : None,
			"health" : None,
		}

	@staticmethod
	def extract_data(path):
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
		return (time, tension)

	@staticmethod
	def compute_bpm(
			data: Optional[dict] = None,
			path: Optional[str] = None,
			remove_first_data: Optional[int] = None,
			param_savgol: Tuple[int] = (200, 500),
			moving_average_window: int = 300,
	) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Compute the BPM from the data
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
		if remove_first_data is not None:
			time = time[remove_first_data:]
			tension = tension[remove_first_data:]

		time_f1, tension_f1 = Oximeter.apply_savgol_filter(time, param_savgol[0]), Oximeter.apply_savgol_filter(tension, param_savgol[0])

		tension_diff = Oximeter.differentiate(tension_f1)

		time_f2, tension_diff_f2 = Oximeter.apply_savgol_filter(time_f1, param_savgol[1]), Oximeter.apply_savgol_filter(tension_diff, param_savgol[1])

		tension_diff_f2_mean = Oximeter.moving_average(tension_diff_f2, moving_average_window)

		bpm = Oximeter.compute_bpm_from_signal(time_f2, tension_diff_f2_mean, moving_average_window)

		return bpm


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
	def moving_average(data_to_average: np.ndarray, window:int = 300):
		return np.convolve(data_to_average, np.ones(window), 'valid') / window

	@staticmethod
	def compute_bpm_from_signal(time: np.ndarray, tension: np.ndarray, offset: int) -> float:
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
		return bpm


if __name__ == '__main__':
	#bpm = Oximeter.compute_bpm(path="data_antho_real_good.npy", remove_first_data=100)