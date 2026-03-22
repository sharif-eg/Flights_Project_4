
import pandas as pd
import numpy as np
import plotly.express as px
from part_1_functions import (
    airports as airports_csv,
    plot_route_from_nyc,
    plot_routes_from_nyc,
    compute_euclidean_to_jfk,
    compute_geo_distance
)


# PART 1 — Getting acquainted with the data
# ===================================================


airports = airports_csv.copy()

# getting to know the data
print("Shape (rows, cols):", airports.shape)
print("\nColumns:\n", airports.columns)
print("\nFirst 5 rows:\n", airports.head())
print("\nMissing values per column:\n", airports.isnull().sum())
print("\nDuplicate rows:", airports.duplicated().sum())

# world map of all airports
fig_world = px.scatter_geo(
    airports,
    lat="lat",
    lon="lon",
    color="alt",
    hover_name="name",
    hover_data=["faa", "alt", "tzone"],
    projection="natural earth",
    title="Airports (world)"
)
fig_world.show()

# identify US vs non-US airports, US-only map colored by altitude
airports_us = airports[airports["is_us"]].copy()
airports_outside_us = airports[~airports["is_us"]].copy()

print("\nUS airports:", len(airports_us))
print("Outside-US airports:", len(airports_outside_us))
print("\nOutside-US FAA codes:\n", airports_outside_us["faa"].tolist())

fig_us = px.scatter_geo(
    airports_us,
    lat="lat",
    lon="lon",
    color="alt",
    hover_name="name",
    hover_data=["faa", "alt", "tzone"],
    projection="albers usa",
    title="Airports in the US (colored by altitude)"
)
fig_us.show()

# plot route from NYC to a single airport
plot_route_from_nyc("LAX")

# plot routes from NYC to multiple airports
plot_routes_from_nyc(["LAX", "ABE"])

# Euclidean distance from JFK + distribution
airports = compute_euclidean_to_jfk(airports)

jfk_row = airports[airports["faa"].str.upper() == "JFK"]
print("\nJFK position (lat, lon):", float(jfk_row.iloc[0]["lat"]), float(jfk_row.iloc[0]["lon"]))

fig_euc = px.histogram(
    airports,
    x="dist_to_jfk",
    nbins=100,
    title="Distribution of Euclidean distances from JFK to all airports",
    labels={"dist_to_jfk": "Euclidean distance (lat/lon units)"}
)
fig_euc.show()

print("\nEuclidean distance summary:")
print(airports["dist_to_jfk"].describe())

# Geodesic distance
airports = compute_geo_distance(airports)

fig_geo = px.histogram(
    airports,
    x="geo_dist_to_ref_km",
    nbins=100,
    title="Histogram: Geodesic distances from JFK to all airports",
    labels={"geo_dist_to_ref_km": "Geodesic distance to JFK (km)"}
)
fig_geo.show()

print("\nGeodesic distance summary (km):")
print(airports["geo_dist_to_ref_km"].describe())

# time zone analysis
tzone_counts = (
    airports["tzone"]
    .dropna()
    .value_counts(normalize=True)
    .sort_index()
    .reset_index()
)
tzone_counts.columns = ["tzone", "relative_share"]

print("\nRelative share of airports by time zone:\n", tzone_counts)

fig_tzone = px.bar(
    tzone_counts,
    x="tzone",
    y="relative_share",
    title="Relative share of airports by time zone",
    labels={"tzone": "Time zone", "relative_share": "Relative share"}
)
fig_tzone.show()

# Extra: average altitude per time zone
alt_by_tzone = airports.dropna(subset=["tzone", "alt"]).groupby("tzone")["alt"].mean().reset_index()

fig_alt = px.bar(
    alt_by_tzone,
    x="tzone",
    y="alt",
    title="Average altitude per time zone",
    labels={"tzone": "Time zone", "alt": "Average altitude"}
)
fig_alt.show()

# Extra: top 10 highest airports
top10 = airports.dropna(subset=["alt"]).sort_values("alt", ascending=False)[["faa", "name", "alt", "tzone"]].head(10)
print("\nTop 10 highest-altitude airports:")
print(top10)