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
)



# EXTRA - Data preparation (same wrangling as Part 4)
# ===============================================================

connection = sqlite3.connect("data/flights_database.db")
flights = pd.read_sql("SELECT * FROM flights", connection)

flights = flights.dropna(subset=['dep_time', 'dep_delay', 'arr_time', 'arr_delay', 'air_time'], how='all')
flights['arr_delay'] = flights.groupby(['origin', 'dest'])['arr_delay'].transform(lambda x: x.fillna(x.mean()))
flights['air_time'] = flights.groupby(['origin', 'dest'])['air_time'].transform(lambda x: x.fillna(x.mean()))

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

flights.loc[flights['dep_time'] < flights['sched_dep_time'] - 1200,
            'dep_time_dt'] = flights['dep_time_dt'] + timedelta(days=1)
flights.loc[flights['arr_time'] < flights['sched_arr_time'] - 1200,
            'arr_time_dt'] = flights['arr_time_dt'] + timedelta(days=1)
flights.loc[flights['arr_time_dt'] < flights['dep_time_dt'],
            'arr_time_dt'] = flights['arr_time_dt'] + timedelta(days=1)

flights['dep_delay'] = (flights['dep_time_dt'] - flights['sched_dep_time_dt']).dt.total_seconds() / 60
flights['arr_delay'] = (flights['arr_time_dt'] - flights['sched_arr_time_dt']).dt.total_seconds() / 60
flights.loc[flights['arr_delay'] > 720, 'arr_delay'] = (
    flights.loc[flights['arr_delay'] > 720, 'arr_delay'] - 1440
)

airports_tz = pd.read_sql("SELECT faa, tz FROM airports", connection)
flights = pd.merge(flights, airports_tz.rename(columns={'faa': 'origin', 'tz': 'tz_origin'}), on='origin', how='left')
flights = pd.merge(flights, airports_tz.rename(columns={'faa': 'dest', 'tz': 'tz_dest'}), on='dest', how='left')
flights['tz_diff'] = flights['tz_dest'] - flights['tz_origin']
flights['arr_time_local'] = flights['arr_time_dt'] + pd.to_timedelta(flights['tz_diff'], unit='h')

airlines = pd.read_sql("SELECT * FROM airlines", connection)
planes = pd.read_sql("SELECT * FROM planes", connection)
weather = pd.read_sql("SELECT * FROM weather", connection)

flights = pd.merge(flights, airlines, on='carrier', how='left')
flights.rename(columns={'name': 'airline_name'}, inplace=True)
flights = pd.merge(flights, planes, on='tailnum', how='left', suffixes=('', '_plane'))
flights = pd.merge(flights, weather, on=['origin', 'year', 'month', 'day', 'hour'], how='left', suffixes=('', '_weather'))



# Extra analysis
# ===================================================

# worst month for delays
avg_delay_month = flights.groupby('month')['arr_delay'].mean().reset_index()
avg_delay_month.columns = ['month', 'avg_arr_delay']
worst_month = avg_delay_month.loc[avg_delay_month['avg_arr_delay'].idxmax()]
print(f"\nWorst month:")
print(f"Month {int(worst_month['month'])} has the highest avg arrival delay: {worst_month['avg_arr_delay']:.2f} min")

fig = px.bar(avg_delay_month, x='month', y='avg_arr_delay',
             title='Average Arrival Delay per Month',
             labels={'month': 'Month', 'avg_arr_delay': 'Average Arrival Delay (min)'})
fig.update_xaxes(dtick=1)
fig.show()

# delays by origin airport
delay_by_origin = flights.groupby('origin').agg(
    avg_dep_delay=('dep_delay', 'mean'),
    avg_arr_delay=('arr_delay', 'mean'),
    n_flights=('flight', 'count')
).reset_index()
print(f"\nDelays by origin airport")
print(delay_by_origin.sort_values('avg_dep_delay', ascending=False))

fig = px.bar(delay_by_origin, x='origin', y=['avg_dep_delay', 'avg_arr_delay'], barmode='group',
             title='Average Delay by NYC Origin Airport',
             labels={'value': 'Average Delay (min)', 'origin': 'Airport'})
fig.show()

# top 10 busiest routes
route_counts = flights.groupby(['origin', 'dest']).agg(
    n_flights=('flight', 'count')
).reset_index().sort_values('n_flights', ascending=False)
top10_routes = route_counts.head(10).copy()
top10_routes['route'] = top10_routes['origin'] + ' -> ' + top10_routes['dest']
print(f"\nTop 10 busiest routes")
print(top10_routes[['route', 'n_flights']])

fig = px.bar(top10_routes, x='route', y='n_flights',
             title='Top 10 Busiest Routes from NYC',
             labels={'route': 'Route', 'n_flights': 'Number of Flights'})
fig.update_layout(xaxis_tickangle=-45)
fig.show()

busiest_dest = top10_routes.iloc[0]['dest']
plot_routes_from_nyc([busiest_dest])

# best and worst on-time airline
airline_performance = flights.groupby('airline_name').agg(
    avg_arr_delay=('arr_delay', 'mean'),
    pct_on_time=('arr_delay', lambda x: (x <= 0).mean() * 100),
    n_flights=('flight', 'count')
).reset_index()
airline_performance = airline_performance[airline_performance['n_flights'] >= 100]

best_airline = airline_performance.loc[airline_performance['avg_arr_delay'].idxmin()]
worst_airline = airline_performance.loc[airline_performance['avg_arr_delay'].idxmax()]
print(f"\nBest on-time airline")
print(f"{best_airline['airline_name']} — avg delay: {best_airline['avg_arr_delay']:.2f} min, on-time: {best_airline['pct_on_time']:.1f}%")
print(f"\nWorst on-time airline")
print(f"{worst_airline['airline_name']} — avg delay: {worst_airline['avg_arr_delay']:.2f} min, on-time: {worst_airline['pct_on_time']:.1f}%")

fig = px.bar(airline_performance.sort_values('avg_arr_delay'), x='airline_name', y='avg_arr_delay',
             title='Average Arrival Delay per Airline (Best -> Worst)',
             labels={'airline_name': 'Airline', 'avg_arr_delay': 'Avg Arrival Delay (min)'})
fig.update_layout(xaxis_tickangle=-45)
fig.show()

# airline weather resilience
flights['bad_weather'] = (flights['precip'] > 0) | (flights['wind_speed'] > 20)

weather_impact = flights.groupby(['airline_name', 'bad_weather'])['arr_delay'].mean().reset_index()
weather_impact = weather_impact.pivot(index='airline_name', columns='bad_weather', values='arr_delay')
weather_impact.columns = ['clear_delay', 'bad_delay']
weather_impact['delay_increase'] = weather_impact['bad_delay'] - weather_impact['clear_delay']
weather_impact = weather_impact.dropna().sort_values('delay_increase')
print(f"\nAirline weather resilience")
print(weather_impact.head(5))

fig = px.bar(weather_impact.reset_index(), x='airline_name', y='delay_increase',
             title='Delay Increase in Bad Weather per Airline',
             labels={'airline_name': 'Airline', 'delay_increase': 'Extra Delay in Bad Weather (min)'})
fig.update_layout(xaxis_tickangle=-45)
fig.show()

# rainy vs clear days
flights['rainy'] = flights['precip'] > 0
rainy_perf = flights.groupby(['airline_name', 'rainy'])['arr_delay'].mean().reset_index()
rainy_perf['weather'] = rainy_perf['rainy'].map({True: 'Rainy', False: 'Clear'})

fig = px.bar(rainy_perf, x='airline_name', y='arr_delay', color='weather', barmode='group',
             title='Airline Arrival Delay: Rainy vs Clear Days',
             labels={'airline_name': 'Airline', 'arr_delay': 'Average Arrival Delay (min)'})
fig.update_layout(xaxis_tickangle=-45)
fig.show()

rainy_pivot = rainy_perf.pivot(index='airline_name', columns='weather', values='arr_delay')
rainy_pivot['rain_penalty'] = rainy_pivot['Rainy'] - rainy_pivot['Clear']
print(f"\nRainy vs clear performance")
print(rainy_pivot.sort_values('rain_penalty'))

# plane age vs weather delay
flights['plane_age'] = 2023 - flights['year_plane']
flights['age_group'] = pd.cut(flights['plane_age'], bins=[0, 5, 10, 15, 20, 50],
                               labels=['0-5 yr', '5-10 yr', '10-15 yr', '15-20 yr', '20+ yr'])

age_weather = flights.dropna(subset=['age_group']).groupby(['age_group', 'bad_weather'])['arr_delay'].mean().reset_index()
age_weather['weather'] = age_weather['bad_weather'].map({True: 'Bad Weather', False: 'Clear'})
print(f"\nPlane age vs weather delay")
age_pivot = age_weather.pivot(index='age_group', columns='weather', values='arr_delay')
age_pivot['weather_penalty'] = age_pivot['Bad Weather'] - age_pivot['Clear']
print(age_pivot)

fig = px.bar(age_weather, x='age_group', y='arr_delay', color='weather', barmode='group',
             title='Plane Age vs Arrival Delay in Clear vs Bad Weather',
             labels={'age_group': 'Plane Age', 'arr_delay': 'Average Arrival Delay (min)'})
fig.show()

# plane size vs wind delay
flights['size_group'] = pd.cut(flights['seats'], bins=[0, 50, 150, 250, 500],
                                labels=['Small (<50)', 'Medium (50-150)', 'Large (150-250)', 'Very Large (>250)'])
flights['windy'] = flights['wind_speed'] > 20

size_wind = flights.dropna(subset=['size_group']).groupby(['size_group', 'windy'])['arr_delay'].mean().reset_index()
size_wind['condition'] = size_wind['windy'].map({True: 'Windy', False: 'Calm'})
print(f"\nPlane size vs wind delay")
size_pivot = size_wind.pivot(index='size_group', columns='condition', values='arr_delay')
size_pivot['wind_penalty'] = size_pivot['Windy'] - size_pivot['Calm']
print(size_pivot)

fig = px.bar(size_wind, x='size_group', y='arr_delay', color='condition', barmode='group',
             title='Plane Size vs Arrival Delay: Windy vs Calm Conditions',
             labels={'size_group': 'Plane Size', 'arr_delay': 'Avg Arrival Delay (min)'})
fig.show()

# airline fleet age
avg_plane_age = flights.dropna(subset=['plane_age']).groupby('airline_name').agg(
    avg_plane_age=('plane_age', 'mean'),
    median_plane_age=('plane_age', 'median'),
    n_flights=('flight', 'count')
).reset_index().sort_values('avg_plane_age')
print(f"\nAirlines by average plane age")
print(avg_plane_age)

fig = px.bar(avg_plane_age, x='airline_name', y='avg_plane_age',
             title='Average Plane Age per Airline (lower = newer fleet)',
             labels={'airline_name': 'Airline', 'avg_plane_age': 'Avg Plane Age (years)'})
fig.update_layout(xaxis_tickangle=-45)
fig.show()



# Final summary
# ===================================================

print("\n\nFinal summary")
print(f"Worst month for delays:          Month {int(worst_month['month'])}")
print(f"Most delayed origin airport:     {delay_by_origin.sort_values('avg_dep_delay', ascending=False).iloc[0]['origin']}")
print(f"Busiest route:                   {top10_routes.iloc[0]['route']} ({int(top10_routes.iloc[0]['n_flights'])} flights)")
print(f"Best on-time airline:            {best_airline['airline_name']}")
print(f"Worst on-time airline:           {worst_airline['airline_name']}")
print(f"Best weather handler:            {weather_impact.index[0]}")
print(f"Newest fleet airline:            {avg_plane_age.iloc[0]['airline_name']}")

connection.close()