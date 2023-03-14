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
	def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Load existing data from a path
		:param path: path to the data
		:return: time and tension data
		"""
		time, tension = Oximeter.extract_data(path)

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

