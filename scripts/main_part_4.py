import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta
from part_1_functions import (
    plot_routes_from_nyc,
)
from part_4_functions import (
    format_time,
    check_data_consistency,
    airport_delay_summary,
    airport_route_summary
)


# PART 4 - Data wrangling
# ===================================================

connection = sqlite3.connect("data/flights_database.db")
flights = pd.read_sql("SELECT * FROM flights", connection)

# check missing values
print(flights.isnull().sum())

print(flights['arr_delay'].describe())
delayed = flights[flights['arr_delay'] > 0].shape[0]
total = flights['arr_delay'].notna().sum()
print(f"Percentage of flights with a delay: {delayed / total * 100:.2f}%")

# drop cancelled flights (all 5 key columns missing)
flights = flights.dropna(subset=['dep_time', 'dep_delay', 'arr_time', 'arr_delay', 'air_time'], how='all')
print(f"\n{flights.isnull().sum()}")

# fill arr_delay and air_time with route means
flights['arr_delay'] = flights.groupby(['origin', 'dest'])['arr_delay'].transform(lambda x: x.fillna(x.mean()))
flights['air_time'] = flights.groupby(['origin', 'dest'])['air_time'].transform(lambda x: x.fillna(x.mean()))

print(f"\n{flights.isnull().sum()}")
print(flights.duplicated(subset=['year','month','day','dep_time','carrier','flight','origin']).sum())
print(flights[['dep_time', 'sched_dep_time', 'arr_time', 'sched_arr_time']].head(10))

# convert to datetime
flights['date_str'] = (flights['year'].astype(str) + '-' +
                       flights['month'].astype(str).str.zfill(2) + '-' +
                       flights['day'].astype(str).str.zfill(2))

for col in ['dep_time', 'sched_dep_time', 'arr_time', 'sched_arr_time']:
    time_str = flights[col].apply(format_time)
    flights[col + '_dt'] = pd.to_datetime(
        flights['date_str'] + ' ' + time_str,
        format='%Y-%m-%d %H%M',
        errors='coerce'
    )

# fill missing arr_time_dt using proper datetime arithmetic
mask_fill_arr_time = (
    flights['arr_time_dt'].isna() &
    flights['sched_arr_time_dt'].notna() &
    flights['arr_delay'].notna()
)
fill_values = (
    flights.loc[mask_fill_arr_time, 'sched_arr_time_dt'] +
    pd.to_timedelta(flights.loc[mask_fill_arr_time, 'arr_delay'], unit='m')
).dt.as_unit(flights['arr_time_dt'].dt.unit)
flights.loc[mask_fill_arr_time, 'arr_time_dt'] = fill_values

flights.drop(columns=['date_str'], inplace=True)

# midnight crossing fixes
flights.loc[flights['dep_time'] < flights['sched_dep_time'] - 1200,
            'dep_time_dt'] = flights['dep_time_dt'] + timedelta(days=1)
flights.loc[flights['arr_time'] < flights['sched_arr_time'] - 1200,
            'arr_time_dt'] = flights['arr_time_dt'] + timedelta(days=1)
flights.loc[flights['arr_time_dt'] < flights['dep_time_dt'],
            'arr_time_dt'] = flights['arr_time_dt'] + timedelta(days=1)

# consistency check before fix
check_data_consistency(flights)

# recalculate delays from datetime columns
flights['dep_delay'] = (flights['dep_time_dt'] - flights['sched_dep_time_dt']).dt.total_seconds() / 60
flights['arr_delay'] = (flights['arr_time_dt'] - flights['sched_arr_time_dt']).dt.total_seconds() / 60
# verify fix
check_data_consistency(flights)

# fix midnight-crossing errors: delays over 12h (720 min) are off by exactly one day
flights.loc[flights['arr_delay'] > 720, 'arr_delay'] = (
    flights.loc[flights['arr_delay'] > 720, 'arr_delay'] - 1440
)



# local arrival time with timezone difference
airports_tz = pd.read_sql("SELECT faa, tz FROM airports", connection)
flights = pd.merge(flights, airports_tz.rename(columns={'faa': 'origin', 'tz': 'tz_origin'}), on='origin', how='left')
flights = pd.merge(flights, airports_tz.rename(columns={'faa': 'dest', 'tz': 'tz_dest'}), on='dest', how='left')
flights['tz_diff'] = flights['tz_dest'] - flights['tz_origin']
flights['arr_time_local'] = flights['arr_time_dt'] + pd.to_timedelta(flights['tz_diff'], unit='h')

print(flights[['origin', 'dest', 'tz_origin', 'tz_dest', 'tz_diff', 'arr_time_dt', 'arr_time_local']].head())

# merge airlines, planes, weather
airlines = pd.read_sql("SELECT * FROM airlines", connection)
planes = pd.read_sql("SELECT * FROM planes", connection)
weather = pd.read_sql("SELECT * FROM weather", connection)

flights = pd.merge(flights, airlines, on='carrier', how='left')
flights.rename(columns={'name': 'airline_name'}, inplace=True)
flights = pd.merge(flights, planes, on='tailnum', how='left', suffixes=('', '_plane'))
flights = pd.merge(flights, weather, on=['origin', 'year', 'month', 'day', 'hour'], how='left', suffixes=('', '_weather'))

# effect of weather on plane types
weather_by_type = flights.groupby('type')[['arr_delay', 'wind_speed', 'precip']].mean()
print("\nAverage arrival delay, wind speed and precipitation per plane type:")
print(weather_by_type.sort_values('arr_delay', ascending=False))

# most delayed airports
delay_by_airport = flights.groupby('dest').agg(
    avg_arr_delay=('arr_delay', 'mean'),
    median_arr_delay=('arr_delay', 'median'),
    n_flights=('flight', 'count')
).sort_values('avg_arr_delay', ascending=False)
print("\nMost delayed destination airports:")
print(delay_by_airport.head(10))

# most common routes
routes = flights.groupby(['origin', 'dest']).agg(
    n_flights=('flight', 'count'),
    avg_arr_delay=('arr_delay', 'mean')
).sort_values('n_flights', ascending=False)
print("\nMost common routes from NYC:")
print(routes.head(10))

# fastest plane models
fastest_planes = flights.dropna(subset=['model', 'air_time', 'distance']).copy()
fastest_planes['minutes_per_mile'] = fastest_planes['air_time'] / fastest_planes['distance']
fastest_plane_summary = fastest_planes.groupby('model').agg(
    avg_minutes_per_mile=('minutes_per_mile', 'mean'),
    n_flights=('flight', 'count')
)
fastest_plane_summary = fastest_plane_summary[fastest_plane_summary['n_flights'] >= 50]
fastest_plane_summary = fastest_plane_summary.sort_values('avg_minutes_per_mile')
print("\nFastest plane models:")
print(fastest_plane_summary.head(10))

# weather vs delay correlation
corr_wind = flights[['arr_delay', 'wind_speed']].corr().iloc[0, 1]
corr_precip = flights[['arr_delay', 'precip']].corr().iloc[0, 1]
print(f"\nCorrelation between wind speed and arrival delay: {corr_wind:.4f}")
print(f"Correlation between precipitation and arrival delay: {corr_precip:.4f}")

# dashboard function examples
print("\nExample summary for JFK:")
print(airport_delay_summary(flights, origin='JFK'))
print("\nExample routes from JFK:")
print(airport_route_summary(flights, origin='JFK').head(10))

connection.close()