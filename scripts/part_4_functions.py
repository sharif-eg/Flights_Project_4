import pandas as pd
import numpy as np
import sqlite3
 
connection = sqlite3.connect("data/flights_database.db")
cursor = connection.cursor()

# convert a numeric time like 517.0 into a string '0517' for datetime parsing. Handles 2400 as midnight (0000).
def format_time(t):
    if pd.isna(t):
        return np.nan
    t = int(t)
    if t == 2400:
        t = 0
    return f"{t:04d}"
 
# check whether dep_delay, arr_delay and air_time are consistent with the datetime columns.
# check if dep_delay matches dep_time_dt - sched_dep_time_dt
def check_data_consistency(flights):
    calc_dep_delay = (flights['dep_time_dt'] - flights['sched_dep_time_dt']).dt.total_seconds() / 60
    dep_mismatch = (calc_dep_delay - flights['dep_delay']).abs() > 1
    print(f"Departure delay mismatches: {dep_mismatch.sum()}")
 
    # check if arr_delay matches arr_time_dt - sched_arr_time_dt
    calc_arr_delay = (flights['arr_time_dt'] - flights['sched_arr_time_dt']).dt.total_seconds() / 60
    arr_mismatch = (calc_arr_delay - flights['arr_delay']).abs() > 1
    print(f"Arrival delay mismatches: {arr_mismatch.sum()}")
 
    # check if elapsed flight time roughly matches air_time
    # this will not be exact because elapsed time includes taxi time
    calc_elapsed = (flights['arr_time_dt'] - flights['dep_time_dt']).dt.total_seconds() / 60
    air_mismatch = (calc_elapsed - flights['air_time']).abs() > 60
    print(f"Large elapsed time vs air_time mismatches: {air_mismatch.sum()}")
 
# return key delay statistics
def airport_delay_summary(df, origin=None, dest=None):
    result = df.copy()
 
    if origin is not None:
        result = result[result['origin'] == origin]
 
    if dest is not None:
        result = result[result['dest'] == dest]
 
    return pd.Series({
        'n_flights': result['flight'].count(),
        'avg_dep_delay': result['dep_delay'].mean(),
        'avg_arr_delay': result['arr_delay'].mean(),
        'median_arr_delay': result['arr_delay'].median(),
        'avg_air_time': result['air_time'].mean(),
        'avg_distance': result['distance'].mean()
    })
 
# return per-route statistics, filtered by origin and destination.
def airport_route_summary(df, origin=None, dest=None):
    result = df.copy()
 
    if origin is not None:
        result = result[result['origin'] == origin]
 
    if dest is not None:
        result = result[result['dest'] == dest]
 
    return result.groupby(['origin', 'dest']).agg(
        n_flights=('flight', 'count'),
        avg_arr_delay=('arr_delay', 'mean'),
        median_arr_delay=('arr_delay', 'median')
    ).sort_values('n_flights', ascending=False)
 