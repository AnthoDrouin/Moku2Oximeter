from data_analysis import Oximeter
from moku.instruments import Datalogger
import moku
import sys
import numpy as np
from tqdm import tqdm

# datalogger = Datalogger("10.248.117.55", force_connect=True)
# patient = {
# 	"name": "Émile J Knystautas",
# 	"age": 21,
# }
# Oximeter(
# 	ip = "10.248.117.55",
# 	patient=patient,
# 	streaming_period=9,
# 	filename="data_2023_03_16",
# 	show_data=True,
# 	remove_first_data=100,
# ).acquisition()

#print(Oximeter.data["SpO2"], Oximeter.data["BPM"], Oximeter.data["health"])

data = np.load("data_antho_real_good.npy", allow_pickle=True).item()
bpm, extra = Oximeter.compute_bpm(data=data, remove_first_data=10)

# sp02 = []
# batch = []
#
# for i in tqdm(range(5, 8000, 20)):
# 	sp02.append(Oximeter.compute_spO2(data=data, remove_first_data=10, param_savgol=(i, 500), batch_size=2005)[0])
# 	batch.append(i)

#print(max(sp02), batch[sp02.index(max(sp02))])

#print(data["BPM"], data["all_BPMs"][-1] - data["BPM"])


#sp02, err = Oximeter.compute_spO2(data=data, remove_first_data=100, param_savgol=(int(-27.5174*data["BPM"] + 4254.2561),None), batch_size=int(149.98488*data["BPM"] - 8183.47294))
#print(sp02, err)
Oximeter.plot_data(
	extra["time"],
	extra["tension_f1"],
	type_graph="final",
	#index_minimum_time=extra["index_minimum_time"],
	#index_minimum=extra["index_minimum"],
	#bpm=data["BPM"],
)