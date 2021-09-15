#%%
import pandas as pd
import numpy as np
import os

from pandas.io.parsers import read_csv 

sensors = ['android.sensor.accelerometer',
 'android.sensor.gyroscope',
 'android.sensor.gravity',
 'android.sensor.orientation',
 'android.sensor.magnetic_field',
 'android.sensor.light',
 'android.sensor.linear_acceleration',
 'sound',
 'speed',
 'activityrecognition',
 'android.sensor.step_detector',
 'android.sensor.tilt_detector',
 'android.sensor.step_counter',
 'android.sensor.proximity',
 'android.sensor.pressure',
 'android.sensor.magnetic_field_uncalibrated',
 'android.sensor.gyroscope_uncalibrated',
 'android.sensor.game_rotation_vector',
 'android.sensor.rotation_vector']

def combine_data():
    files = os.listdir("raw_data//")
    data = pd.DataFrame()

    progress = 0
    for file in files: 
        for csv in os.listdir("raw_data//" + file):
            try:
                user = csv.split("_")[1]
                transportation= csv.split("_")[2]
                time_stamp = csv.split("_")[3].replace(".csv","")

                df = pd.read_csv("raw_data/" + file + "/"+ csv, error_bad_lines=False, header = None,warn_bad_lines =False)
                df[len(df.columns)] = user
                df[len(df.columns)] = transportation
                df[len(df.columns)] = time_stamp
                df[len(df.columns)] = csv

                data = pd.concat([data,df],ignore_index=False)
                progress += 1
            except:
                pass
            print(f"{(progress/263)*100 :.2f}" )

    data.to_csv("raw_data_combined.csv")

combine_data()
# %%
data = read_csv("raw_data_combined.csv")
data = data.iloc[:,1:10]

data.columns = ["time","sensor","x","y","z","user","moving","timestampbeg","file"]

data = data[data.sensor.isin(sensors)]
data = data[data.moving.isin(['Bus','Car','Still','Train','Walking'])]
data.timestampbeg = data.timestampbeg.apply(lambda x: float(x))
data["datatype"] = data.time.apply(lambda x : type(x))
data = data[(data["datatype"] == int ) | (data["datatype"] == float)]
data["timestamp"] = data["timestampbeg"] + data["time"]
data.drop(["time","timestampbeg","datatype"], axis = 1, inplace=True)
data = data.sort_values(by=["timestamp"],ascending=True).dropna()

# %%