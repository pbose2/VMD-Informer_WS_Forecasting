import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import glob
import os

LIDAR_DATA_PATH = os.path.join(os.getcwd(),"data_LIDAR")
BUOY_DATA_PATH = os.path.join(os.getcwd(),"data_BUOY")
OUTPUT_WSPD_WDIR = "wind_data_cleaned.csv"
OUTPUT_MET = "met_data_merged.csv"

if not os.path.isdir(LIDAR_DATA_PATH):
    raise FileNotFoundError(f"LIDAR data folder not found: {LIDAR_DATA_PATH}")
if not os.path.isdir(BUOY_DATA_PATH):
    raise FileNotFoundError(f"Buoy data folder not found: {BUOY_DATA_PATH}")

data_path = LIDAR_DATA_PATH
files = glob.glob(data_path+"/*.nc")
files = sorted(files)

if not files:
    raise FileNotFoundError(f"No .nc files found in {LIDAR_DATA_PATH}")
print(f"{len(files)} files loaded")


timestamps_list = []
wspd_list = []
wdir = []
for file in files:
    with Dataset(file) as data:
        ts = data.variables['base_time'][:] + data.variables['time_offset'][:]
        wspd_now = data.variables['wspd'][:]
        wdir_now = data.variables['wdir'][:]

        wdir.append(wdir_now)
        timestamps_list.append(ts)
        wspd_list.append(wspd_now)


df = pd.DataFrame({
    'timestamps': np.concatenate(timestamps_list),
    "wspd40": np.concatenate(np.array(wspd_list)[:,0,:]),
    "wspd60": np.concatenate(np.array(wspd_list)[:,1,:]),
    "wspd80": np.concatenate(np.array(wspd_list)[:,2,:]),
    "wspd90": np.concatenate(np.array(wspd_list)[:,3,:]),
    "wspd100": np.concatenate(np.array(wspd_list)[:,4,:]),
    "wspd120": np.concatenate(np.array(wspd_list)[:,5,:]),
    "wspd140": np.concatenate(np.array(wspd_list)[:,6,:]),

    "wdir40": np.concatenate(np.array(wdir)[:,0,:]),
    "wdir60": np.concatenate(np.array(wdir)[:,1,:]),
    "wdir80": np.concatenate(np.array(wdir)[:,2,:]),
    "wdir90": np.concatenate(np.array(wdir)[:,3,:]),
    "wdir100": np.concatenate(np.array(wdir)[:,4,:]),
    "wdir120": np.concatenate(np.array(wdir)[:,5,:]),
    "wdir140": np.concatenate(np.array(wdir)[:,6,:])
})

df['timestamps'] = pd.to_datetime(df['timestamps'],unit = 's')

for column in df.drop(['timestamps'],axis=1):
    idx = df[column] == -9999
    print(f'number of missing datapoint at {column}: ',idx.sum())

for column in df.drop(['timestamps'], axis=1):
    arr = np.array(df[column])
    for i, val in enumerate(arr):
        if val == -9999 and i >= 3:
            arr[i] = (arr[i-1] + arr[i-2] + arr[i-3]) / 3
    df[column] = arr

print("\nAfter removing NaNs\n")
for column in df.drop(['timestamps'],axis=1):
    idx = df[column] == -9999
    print(f'number of missing datapoint at {column}: ',idx.sum())


times = []
pressure = []
rh = []
air_temp = []
sea_temp = []
wspd_4 = []

files_10m = glob.glob(BUOY_DATA_PATH+"/*10m.a2e.nc")
files_10m = sorted(files_10m)

if not files_10m:
    raise FileNotFoundError(f"No buoy (*10m.a2e.nc) files found in {BUOY_DATA_PATH}")

for file in files_10m:
    with Dataset(file) as data:
        pressure_now = data.variables['pressure'][:]
        times_now = data.variables['time'][:]
        rh_now = data.variables['rh'][:]
        air_temp_now = data.variables['air_temperature'][:]
        sea_temp_now = data.variables['YSI_SST'][:]
        wspd_4_now = data.variables['wind_speed'][:]

        times.append(times_now)
        pressure.append(pressure_now)
        rh.append(rh_now)
        air_temp.append(air_temp_now)
        sea_temp.append(sea_temp_now)
        wspd_4.append(wspd_4_now)


df_buoy = pd.DataFrame({
    'time': np.concatenate(times),
    'pressure': np.concatenate(pressure),
    'rh': np.concatenate(rh),
    'air temp': np.concatenate(air_temp),
    'sea temp': np.concatenate(sea_temp),
    'wspd': np.concatenate(wspd_4)
    # ,'dT': np.concatenate(np.array(air_temp)) - np.concatenate(np.array(sea_temp))
})

df_buoy['time'] = pd.to_datetime(df_buoy['time'],unit='s')

for column in df_buoy.columns:
    print(column,len(df_buoy[column][df_buoy[column]!=-9999]))

cols = df_buoy.columns
df_buoy[cols] = df_buoy[cols].replace(-9999, np.nan)
df_buoy.index = df_buoy['time']

for column in df_buoy.columns:
    print(column, df_buoy[column].notna().sum())

df_buoy['sea temp'] = df_buoy['sea temp'].interpolate(method='time',limit=3)
df_buoy['pressure'] = df_buoy['pressure'].interpolate(method='time',limit=3)
df_buoy['air temp'] = df_buoy['air temp'].interpolate(method='time',limit=3)
df_buoy['wspd'] = df_buoy['wspd'].interpolate(method='time',limit=3)
df_buoy['rh'] = df_buoy['rh'].interpolate(method='time',limit=3)
df_buoy.dropna(inplace=True)

df.index=df['timestamps']

df_merged = df_buoy.join(df, how='inner',lsuffix='_buoy',rsuffix='_lidar')
df_merged['time_diff'] = df_merged.index.diff().total_seconds()/60
df_merged.dropna(inplace=True)

df[['timestamps'] + [c for c in df.columns if c != 'timestamps']].to_csv(OUTPUT_WSPD_WDIR, index=False)
df_merged.to_csv(OUTPUT_MET)
