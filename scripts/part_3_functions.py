import pandas as pd
import numpy as np
import plotly.express as px
from scripts.part_1_functions import plot_routes_from_nyc
 
# plot all destinations from a NYC airport on a given day
def plot_flights_from_nyc_day(connection, month, day, origin_airport):
    cursor = connection.cursor()
    query = """
    SELECT dest
    FROM flights
    WHERE month = ? AND day = ? AND origin = ?
    """
    cursor.execute(query, (month, day, origin_airport.upper()))
    rows = cursor.fetchall()
 
    if not rows:
        print(f"No flights from {origin_airport.upper()} on {month}/{day}.")
        return
 
    destinations = list({r[0] for r in rows})  # remove duplicates
    plot_routes_from_nyc(destinations)

# return flight statistics for a given day and origin airport
def flight_statistics_for_day(connection, month, day, origin_airport):
    cursor = connection.cursor()
    query = """
    SELECT dest, distance, sched_dep_time
    FROM flights
    WHERE month = ? AND day = ? AND origin = ?
    """
    cursor.execute(query, (month, day, origin_airport.upper()))
    rows = cursor.fetchall()
    df_day = pd.DataFrame(rows, columns=["dest", "distance", "sched_dep_time"])
 
    n_flights = len(df_day)
    n_unique_destinations = df_day["dest"].nunique()
 
    dest_counts = df_day["dest"].value_counts()
    most_visited_destination = dest_counts.index[0]
    most_visited_count = int(dest_counts.iloc[0])
 
    furthest_dest = df_day.loc[df_day["distance"].idxmax()]
    shortest_dest = df_day.loc[df_day["distance"].idxmin()]
 
    first_flight = df_day.loc[df_day["sched_dep_time"].idxmin()]
    last_flight = df_day.loc[df_day["sched_dep_time"].idxmax()]
 
    return {
        "origin": origin_airport.upper(),
        "month": month,
        "day": day,
        "n_flights": n_flights,
        "n_unique_destinations": n_unique_destinations,
        "most_visited_destination": most_visited_destination,
        "most_visited_count": most_visited_count,
        "furthest_destination": furthest_dest["dest"],
        "furthest_distance": float(furthest_dest["distance"]),
        "shortest_destination": shortest_dest["dest"],
        "shortest_distance": float(shortest_dest["distance"]),
        "first_flight_destination": first_flight["dest"],
        "first_flight_time": int(first_flight["sched_dep_time"]),
        "last_flight_destination": last_flight["dest"],
        "last_flight_time": int(last_flight["sched_dep_time"])
    }
 
# return a dict counting how many times each plane type was used on a route
def plane_types_for_route(connection, origin, dest):
    cursor = connection.cursor()
    query = """
    SELECT tailnum
    FROM flights
    WHERE origin = ? AND dest = ?
    """
    cursor.execute(query, (origin.upper(), dest.upper()))
    rows = cursor.fetchall()
    df_flights = pd.DataFrame(rows, columns=["tailnum"])
 
    query = "SELECT tailnum, type FROM planes"
    cursor.execute(query)
    rows = cursor.fetchall()
    df_planes = pd.DataFrame(rows, columns=["tailnum", "type"])
 
    df_merged = df_flights.merge(df_planes, on="tailnum", how="inner")
    type_counts = df_merged["type"].value_counts()
    return type_counts.to_dict()
 
# return the number of delayed flights to a destination in a month range
def amount_delayed_flights(connection, month_start, month_end, dest):
    cursor = connection.cursor()
    query = """
    SELECT dep_delay
    FROM flights
    WHERE month >= ? AND month <= ? AND dest = ?
    """
    cursor.execute(query, (month_start, month_end, dest.upper()))
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=["dep_delay"])
    df = df.dropna(subset=["dep_delay"])
    delayed = df[df["dep_delay"] > 0]
    return len(delayed)
 
# return the top 5 airplane manufacturers flying to a destination
def top5_manufacturers(connection, dest):
    cursor = connection.cursor()
    query = """
    SELECT tailnum
    FROM flights
    WHERE dest = ?
    """
    cursor.execute(query, (dest,))
    rows = cursor.fetchall()
    df_flights = pd.DataFrame(rows, columns=["tailnum"])
 
    query = "SELECT tailnum, manufacturer FROM planes"
    cursor.execute(query)
    rows = cursor.fetchall()
    df_planes = pd.DataFrame(rows, columns=["tailnum", "manufacturer"])
 
    df_merged = df_flights.merge(df_planes, on="tailnum")
    top5 = df_merged["manufacturer"].value_counts().head(5)
    return top5
 
# compute the inner product between the flight direction vector and wind vector
# positive = tailwind (wind pushes plane forward)
# negative = headwind (wind pushes against the plane)
def compute_inner_product(flight_dir, wind_dir, wind_speed):
    wind_x = wind_speed * np.sin(np.radians(wind_dir))
    wind_y = wind_speed * np.cos(np.radians(wind_dir))
    flight_x = np.sin(np.radians(flight_dir))
    flight_y = np.cos(np.radians(flight_dir))
    return (wind_x * flight_x) + (wind_y * flight_y)
 
# label a flight as tailwind, headwind, or neutral based on inner product sign
def classify_wind(x):
    if x > 0:
        return "Tailwind (+)"
    elif x < 0:
        return "Headwind (-)"
    else:
        return "Neutral (0)"