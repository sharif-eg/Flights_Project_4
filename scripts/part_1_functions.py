import pandas as pd
import plotly.express as px
import numpy as np

airports = pd.read_csv("data/airports.csv")

us_timezones = [
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Los_Angeles",
    "America/Phoenix",
    "America/Anchorage",
    "America/Adak",
    "Pacific/Honolulu",
]
airports["is_us"] = airports["tzone"].isin(us_timezones)


def plot_route_from_nyc(faa_code):
    NYC_LAT = 40.7128
    NYC_LON = -74.0060
    faa_code = faa_code.strip().upper()
    row = airports[airports["faa"] == faa_code]
    if row.empty:
        print(f"FAA code '{faa_code}' not found.")
        return

    target_lat = float(row.iloc[0]["lat"])
    target_lon = float(row.iloc[0]["lon"])
    target_name = row.iloc[0]["name"]
    target_is_us = bool(row.iloc[0]["is_us"])

    fig = px.line_geo(
        lat=[NYC_LAT, target_lat],
        lon=[NYC_LON, target_lon],
        title=f"NYC -> {faa_code} ({target_name})"
    )

    fig.add_scattergeo(
        lat=[NYC_LAT, target_lat],
        lon=[NYC_LON, target_lon],
        mode="markers+text",
        text=["NYC", target_name],
        textposition="top center",
        name="Airports"
    )

    if target_is_us:
        fig.update_geos(scope="usa")

    fig.show()


def plot_routes_from_nyc(faa_codes):
    NYC_LAT = 40.7128
    NYC_LON = -74.0060

    faa_codes = [c.strip().upper() for c in faa_codes]

    targets = []
    for code in faa_codes:
        row = airports[airports["faa"] == code]
        if row.empty:
            print(f"FAA code '{code}' not found. Skipped.")
            continue

        targets.append({
            "faa": code,
            "lat": float(row.iloc[0]["lat"]),
            "lon": float(row.iloc[0]["lon"]),
            "name": row.iloc[0]["name"],
            "is_us": bool(row.iloc[0]["is_us"])
        })

    if len(targets) == 0:
        print("No valid FAA codes provided.")
        return

    all_us = all(t["is_us"] for t in targets)

    fig = px.line_geo(title="NYC -> Multiple Airports")

    for t in targets:
        fig.add_scattergeo(
            lat=[NYC_LAT, t["lat"]],
            lon=[NYC_LON, t["lon"]],
            mode="lines",
            name=f"NYC -> {t['faa']}"
        )

        fig.add_scattergeo(
            lat=[t["lat"]],
            lon=[t["lon"]],
            mode="markers+text",
            text=[t["faa"]],
            textposition="top center",
            showlegend=False
        )

    fig.add_scattergeo(
        lat=[NYC_LAT],
        lon=[NYC_LON],
        mode="markers+text",
        text=["NYC"],
        textposition="top center",
        name="NYC"
    )

    if all_us:
        fig.update_geos(scope="usa")

    fig.show()


def compute_euclidean_to_jfk(df):
    JFK_LAT = 40.63980103
    JFK_LON = -73.77890015
    df["dist_to_jfk"] = np.sqrt((df["lat"] - JFK_LAT) ** 2 + (df["lon"] - JFK_LON) ** 2)
    return df


def compute_geo_distance(df_airports, ref_lat=40.63980103, ref_lon=-73.77890015, r=6371.0):
    lat1 = np.radians(ref_lat)
    lon1 = np.radians(ref_lon)

    lat2 = np.radians(df_airports["lat"].astype(float))
    lon2 = np.radians(df_airports["lon"].astype(float))

    dphi = lat2 - lat1
    dlambda = lon2 - lon1
    phi_m = (lat1 + lat2) / 2

    part1 = (2 * np.sin(dphi / 2) * np.cos(dlambda / 2)) ** 2
    part2 = (2 * np.cos(phi_m) * np.sin(dlambda / 2)) ** 2

    df_airports["geo_dist_to_ref_km"] = r * np.sqrt(part1 + part2)

    return df_airports