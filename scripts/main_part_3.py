import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
from scripts.part_1_functions import (
    airports as airports_csv,
    plot_routes_from_nyc,
    compute_euclidean_to_jfk,
    compute_geo_distance
)
from scripts.part_3_functions import (
    plot_flights_from_nyc_day,
    flight_statistics_for_day,
    plane_types_for_route,
    amount_delayed_flights,
    top5_manufacturers,
    compute_inner_product,
    classify_wind
)


# PART 3 — Interacting with the database
# ===================================================

connection = sqlite3.connect("flights_database.db")
cursor = connection.cursor()

# load JFK flights for distance verification
query = "SELECT origin, dest, distance FROM flights WHERE origin = 'JFK'"
cursor.execute(query)
rows = cursor.fetchall()
df = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])

df_unique_dest = df.drop_duplicates(subset="dest", keep="first")

# load airports from database
query = "SELECT faa, name, lat, lon, alt, tzone FROM airports"
cursor = connection.cursor()
cursor.execute(query)
rows = cursor.fetchall()
airports_db = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])

# JFK coordinates from DB
jfk_row = airports_db[airports_db["faa"].str.upper() == "JFK"]
JFK_LAT = float(jfk_row.iloc[0]["lat"])
JFK_LON = float(jfk_row.iloc[0]["lon"])
print("\nJFK position (lat, lon):", JFK_LAT, JFK_LON)

# compute distances using Part 1 functions
airports_db = compute_euclidean_to_jfk(airports_db)
airports_db = compute_geo_distance(airports_db)
airports_db["geo_dist_to_ref_miles"] = airports_db["geo_dist_to_ref_km"] * 0.621371

# verify distances match the database
df_compare = df_unique_dest.merge(
    airports_db[["faa", "geo_dist_to_ref_miles"]],
    left_on="dest",
    right_on="faa",
    how="left"
)

fig = px.scatter(
    df_compare,
    x="distance",
    y="geo_dist_to_ref_miles",
    hover_data=["dest"],
    labels={
        "distance": "Database distance (miles)",
        "geo_dist_to_ref_miles": "Computed geodesic distance (miles)"
    },
    title="Database distance vs Computed geodesic distance from JFK"
)
fig.show()

# identify all NYC origin airports
cursor = connection.cursor()
cursor.execute("SELECT DISTINCT origin FROM flights")
nyc_origins = [r[0] for r in cursor.fetchall()]
print("\nNYC origin airports:", nyc_origins)

df_nyc_airports = airports_db[airports_db["faa"].isin(nyc_origins)]
print("\nNYC airport info:")
print(df_nyc_airports)

# plot destinations from JFK on Feb 18
plot_flights_from_nyc_day(connection, 2, 18, "JFK")

# flight statistics for that day
print(flight_statistics_for_day(connection, 2, 18, "JFK"))

# plane types on JFK -> BOS
print("Plane types used on JFK -> BOS:")
print(plane_types_for_route(connection, "JFK", "BOS"))

# average departure delay per airline (barplot with rotated names)
cursor.execute("SELECT carrier, dep_delay FROM flights")
rows = cursor.fetchall()
df_delays = pd.DataFrame(rows, columns=["carrier", "dep_delay"])
df_delays = df_delays.dropna(subset=["dep_delay"])

cursor.execute("SELECT carrier, name FROM airlines")
rows = cursor.fetchall()
df_airlines = pd.DataFrame(rows, columns=["carrier", "name"])
df_merged = df_delays.merge(df_airlines, on="carrier")
avg_delay = df_merged.groupby("name")["dep_delay"].mean().reset_index()

fig = px.bar(
    avg_delay,
    x="name",
    y="dep_delay",
    title="Average Departure Delay per Airline",
    labels={"name": "Airline", "dep_delay": "Average Departure Delay (min)"}
)
overall_avg = df_merged["dep_delay"].mean()
fig.add_hline(
    y=overall_avg,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Overall average: {overall_avg:.1f} min",
    annotation_position="top right"
)
fig.update_layout(xaxis_tickangle=-45)
fig.show()

# delayed flights to LAX from January to June
print(amount_delayed_flights(connection, 1, 6, "LAX"))

# top 5 manufacturers flying to LAX
print(top5_manufacturers(connection, "LAX"))

# extra: top 5 manufacturers overall
cursor = connection.cursor()
cursor.execute("SELECT tailnum FROM flights")
rows = cursor.fetchall()
df_flights = pd.DataFrame(rows, columns=["tailnum"])
cursor.execute("SELECT tailnum, manufacturer FROM planes")
rows = cursor.fetchall()
df_planes = pd.DataFrame(rows, columns=["tailnum", "manufacturer"])
df_merged = df_flights.merge(df_planes, on="tailnum")
top5 = df_merged["manufacturer"].value_counts().head(5)
fig = px.bar(
    top5,
    title="Top 5 Airplane Manufacturers (by number of flights)",
    labels={"value": "Number of Flights", "index": "Manufacturer"}
)
fig.show()

# distance vs arrival delay
cursor.execute("SELECT distance, arr_delay FROM flights")
rows = cursor.fetchall()
df_dist_delay = pd.DataFrame(rows, columns=["distance", "arr_delay"])
df_dist_delay = df_dist_delay.dropna()
avg_per_dest = df_dist_delay.groupby("distance")["arr_delay"].mean().reset_index()

fig = px.scatter(
    avg_per_dest,
    x="distance",
    y="arr_delay",
    title="Average Arrival Delay vs Flight Distance",
    labels={"distance": "Distance (miles)", "arr_delay": "Average Arrival Delay (min)"}
)
fig.show()
print("Correlation:", df_dist_delay["distance"].corr(df_dist_delay["arr_delay"]))

# compute average speed per model and fill speed column in planes
cursor = connection.cursor()
cursor.execute("SELECT tailnum, distance, air_time FROM flights")
rows = cursor.fetchall()
df_flights = pd.DataFrame(rows, columns=["tailnum", "distance", "air_time"])
df_flights = df_flights.dropna()
df_flights["speed"] = (df_flights["distance"] / df_flights["air_time"]) * 60

cursor.execute("SELECT tailnum, model FROM planes")
rows = cursor.fetchall()
df_planes = pd.DataFrame(rows, columns=["tailnum", "model"])

df_flights = df_flights.merge(df_planes, on="tailnum", how="inner")
avg_speed_per_model = df_flights.groupby("model")["speed"].mean().reset_index()
avg_speed_per_model.columns = ["model", "avg_speed"]

df_planes_updated = df_planes.merge(avg_speed_per_model, on="model", how="left")
for _, row in df_planes_updated.dropna(subset=["avg_speed"]).iterrows():
    cursor.execute(
        "UPDATE planes SET speed = ? WHERE tailnum = ?",
        (float(row["avg_speed"]), row["tailnum"])
    )
connection.commit()

# flight direction from NYC to each airport
cursor = connection.cursor()
cursor.execute("""
SELECT faa, lat, lon FROM airports
WHERE faa IS NOT NULL AND lat IS NOT NULL AND lon IS NOT NULL
""")
rows = cursor.fetchall()
df_airports = pd.DataFrame(rows, columns=["faa", "lat", "lon"])

NYC_LAT = 40.7128
NYC_LON = -74.0060
delta_lon = np.radians(df_airports["lon"] - NYC_LON)
delta_lat = np.radians(df_airports["lat"] - NYC_LAT)
direction = np.degrees(np.arctan2(delta_lon, delta_lat)) % 360
df_airports["flight_direction"] = direction
print(df_airports[["faa", "flight_direction"]].head(10))

# inner product analysis: wind vs flight direction
cursor = connection.cursor()
cursor.execute("""
SELECT origin, dest, air_time, hour, month, day FROM flights
WHERE origin IS NOT NULL AND dest IS NOT NULL AND air_time IS NOT NULL
  AND hour IS NOT NULL AND month IS NOT NULL AND day IS NOT NULL
""")
rows = cursor.fetchall()
df_flights = pd.DataFrame(rows, columns=["origin", "dest", "air_time", "hour", "month", "day"])

cursor.execute("""
SELECT origin, month, day, hour, wind_dir, wind_speed FROM weather
WHERE origin IS NOT NULL AND month IS NOT NULL AND day IS NOT NULL
  AND hour IS NOT NULL AND wind_dir IS NOT NULL AND wind_speed IS NOT NULL
""")
rows = cursor.fetchall()
df_weather = pd.DataFrame(rows, columns=["origin", "month", "day", "hour", "wind_dir", "wind_speed"])

df_merged = df_flights.merge(df_weather, on=["origin", "month", "day", "hour"], how="inner")
df_merged = df_merged.merge(df_airports[["faa", "flight_direction"]], left_on="dest", right_on="faa", how="inner")

df_merged["inner_product"] = compute_inner_product(
    df_merged["flight_direction"], df_merged["wind_dir"], df_merged["wind_speed"]
)
df_merged = df_merged.dropna(subset=["inner_product", "air_time"]).copy()
df_merged["wind_type"] = df_merged["inner_product"].apply(classify_wind)

avg_air_time = df_merged.groupby("wind_type")["air_time"].mean().reset_index()
print("\nAverage air time by wind type:")
print(avg_air_time)

fig = px.bar(
    avg_air_time, x="wind_type", y="air_time",
    title="Average Air Time: Tailwind vs Headwind",
    labels={"wind_type": "Wind Type", "air_time": "Average Air Time (min)"}
)
fig.show()

df_merged["inner_bin"] = pd.cut(df_merged["inner_product"], bins=20)
binned_summary = (
    df_merged.groupby("inner_bin")["air_time"]
    .agg(["median", "mean", "count"]).reset_index()
    .rename(columns={"median": "median_air_time", "mean": "mean_air_time", "count": "n_flights"})
)
binned_summary["inner_bin_label"] = binned_summary["inner_bin"].astype(str)

fig2 = px.line(
    binned_summary, x="inner_bin_label", y="median_air_time",
    title="Median Air Time by Wind Alignment Bin",
    labels={"inner_bin_label": "Inner Product Bin", "median_air_time": "Median Air Time (min)"}
)
fig2.update_layout(xaxis_tickangle=45)
fig2.show()

df_sample = df_merged.sample(3000)
fig3 = px.scatter(
    df_sample, x="inner_product", y="air_time",
    title="Sample of Flights: Inner Product vs Air Time",
    labels={"inner_product": "Inner Product (wind alignment)", "air_time": "Air Time (min)"}
)
fig3.show()

corr = df_merged["inner_product"].corr(df_merged["air_time"])
print(f"\nCorrelation between inner product and air time: {corr:.4f}")
if corr < 0:
    print("A negative correlation means: tailwind flights tend to have shorter air times.")
else:
    print("The positive correlation suggests distance is a confounding variable —")
    print("tailwind flights tend to be longer routes, masking the wind effect.")
print("\nBinned air time summary:")
print(binned_summary[["inner_bin_label", "median_air_time", "mean_air_time", "n_flights"]])

connection.close()