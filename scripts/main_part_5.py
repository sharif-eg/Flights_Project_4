import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# reuse functions from earlier parts of the project
from part_3_functions import (
    classify_wind,
    compute_inner_product,
    top5_manufacturers,
    flight_statistics_for_day,
    amount_delayed_flights
)
from part_4_functions import (
    airport_delay_summary,
    airport_route_summary
)


# PART 5 — Creating a dashboard
# ===================================================

st.set_page_config(
    page_title="NYC Flights Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_data():
    conn = sqlite3.connect("data/flights_database.db")
    flights = pd.read_sql("SELECT * FROM flights", conn)
    airlines = pd.read_sql("SELECT * FROM airlines", conn)
    airports = pd.read_sql("SELECT * FROM airports", conn)
    planes = pd.read_sql("SELECT * FROM planes", conn)
    weather = pd.read_sql("SELECT * FROM weather", conn)
    conn.close()

    us_timezones = [
        "America/New_York", "America/Chicago", "America/Denver",
        "America/Los_Angeles", "America/Phoenix", "America/Anchorage",
        "America/Adak", "Pacific/Honolulu",
    ]
    airports["is_us"] = airports["tzone"].isin(us_timezones)

    flights = flights.dropna(
        subset=["dep_time", "dep_delay", "arr_time", "arr_delay", "air_time"],
        how="all"
    )
    flights["arr_delay"] = flights.groupby(["origin", "dest"])["arr_delay"].transform(
        lambda x: x.fillna(x.mean())
    )
    flights["air_time"] = flights.groupby(["origin", "dest"])["air_time"].transform(
        lambda x: x.fillna(x.mean())
    )
    flights = flights[flights["dep_delay"].between(-60, 720)]

    flights = flights.merge(airlines[["carrier", "name"]], on="carrier", how="left")
    flights.rename(columns={"name": "airline_name"}, inplace=True)

    planes = planes.rename(columns={"year": "plane_year"})
    flights = flights.merge(planes, on="tailnum", how="left", suffixes=("", "_plane"))
    flights = flights.merge(
        weather, on=["origin", "year", "month", "day", "hour"],
        how="left", suffixes=("", "_weather")
    )

    flights["date"] = pd.to_datetime(flights[["year", "month", "day"]], errors="coerce")
    flights["rainy"] = flights["precip"] > 0
    flights["bad_weather"] = (flights["precip"] > 0) | (flights["wind_speed"] > 20)
    flights["on_time"] = flights["arr_delay"] <= 0
    flights["on_time_dep"] = flights["dep_delay"] <= 0
    flights["plane_age"] = flights["year"] - flights["plane_year"]

    return flights, airports


with st.spinner("Loading flight data..."):
    flights, airports = load_data()


# cached figure helpers
# ===================================================

@st.cache_data
def make_correlation_heatmap(df):
    numeric_cols = df[["dep_delay", "arr_delay", "air_time", "distance",
                       "wind_speed", "precip", "visib", "temp"]].dropna()
    if len(numeric_cols) < 10:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Between Delay Factors")
    plt.tight_layout()
    return fig


@st.cache_data
def make_route_map(origin_code, dest_code, airports_df):
    origin_row = airports_df[airports_df["faa"] == origin_code]
    dest_row = airports_df[airports_df["faa"] == dest_code]
    if len(origin_row) == 0 or len(dest_row) == 0:
        return None

    o_lat = float(origin_row.iloc[0]["lat"])
    o_lon = float(origin_row.iloc[0]["lon"])
    d_lat = float(dest_row.iloc[0]["lat"])
    d_lon = float(dest_row.iloc[0]["lon"])
    d_name = str(dest_row.iloc[0]["name"])
    all_us = bool(origin_row.iloc[0]["is_us"]) and bool(dest_row.iloc[0]["is_us"])

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=[o_lat, d_lat], lon=[o_lon, d_lon],
        mode="lines", line=dict(width=2, color="blue"),
        name=f"{origin_code} -> {dest_code}"
    ))
    fig.add_trace(go.Scattergeo(
        lat=[o_lat], lon=[o_lon], mode="markers+text",
        marker=dict(size=10, color="green"),
        text=[origin_code], textposition="top center", name="Origin"
    ))
    fig.add_trace(go.Scattergeo(
        lat=[d_lat], lon=[d_lon], mode="markers+text",
        marker=dict(size=10, color="red"),
        text=[d_name], textposition="top center", name="Destination"
    ))
    if all_us:
        fig.update_geos(scope="usa")
    fig.update_layout(title=f"Route Map: {origin_code} -> {dest_code}", showlegend=True, height=450)
    return fig


# Sidebar
# ===================================================

with st.sidebar:
    st.header("Filters")
    page = st.radio(
        "Sections",
        ["Overview", "Route Analysis", "Delay Analysis",
         "Daily Statistics", "Fleet & Airlines", "Extra Insights"]
    )
    origin_list = sorted(flights["origin"].dropna().unique())
    dest_list = sorted(flights["dest"].dropna().unique())
    selected_origin = st.selectbox("Departure airport", ["All"] + origin_list)
    selected_dest = st.selectbox("Arrival airport", ["All"] + dest_list)

df = flights.copy()
if selected_origin != "All":
    df = df[df["origin"] == selected_origin]
if selected_dest != "All":
    df = df[df["dest"] == selected_dest]


# Overview
# ===================================================

if page == "Overview":
    st.title("NYC Flights Dashboard")
    st.markdown("General statistics of flights departing from NYC in the dataset.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Flights", f"{len(df):,}")
    col2.metric("Average Departure Delay", f"{df['dep_delay'].mean():.1f} min")
    col3.metric("Average Arrival Delay", f"{df['arr_delay'].mean():.1f} min")
    col4.metric("On-Time Departures", f"{(df['on_time_dep'].mean() * 100):.1f}%")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("On-Time Arrivals", f"{(df['on_time'].mean() * 100):.1f}%")
    col6.metric("Unique Destinations", f"{df['dest'].nunique()}")
    col7.metric("Airlines", f"{df['airline_name'].nunique()}")
    col8.metric("Average Distance", f"{df['distance'].mean():.0f} miles")

    # average departure delay per airline
    avg_delay_airline = (
        df.groupby("airline_name")["dep_delay"].mean()
        .reset_index().sort_values("dep_delay")
    )
    fig = px.bar(avg_delay_airline, x="airline_name", y="dep_delay",
                 title="Average Departure Delay per Airline",
                 labels={"airline_name": "Airline", "dep_delay": "Average Delay (min)"})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # top 10 busiest routes
    route_counts = (
        df.groupby(["origin", "dest"]).agg(n=("flight", "count"))
        .reset_index().sort_values("n", ascending=False).head(10)
    )
    route_counts["route"] = route_counts["origin"] + " -> " + route_counts["dest"]
    fig = px.bar(route_counts, x="route", y="n", title="Top 10 Busiest Routes",
                 labels={"route": "Route", "n": "Number of Flights"})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # top 10 airlines by flight count as pie chart
    top_airlines = (
        df.groupby("airline_name").agg(
            n=("flight", "count"), avg_dep=("dep_delay", "mean"),
            median_dep=("dep_delay", "median"), avg_arr=("arr_delay", "mean")
        ).reset_index().sort_values("n", ascending=False).head(10).round(1)
    )
    fig = px.pie(top_airlines, values="n", names="airline_name",
                 title="Top 10 Airlines by Number of Flights")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(top_airlines, use_container_width=True)

    st.divider()

    # per-route summary using airport_route_summary from Part 4
    st.subheader("Top Routes Summary")
    route_summary = airport_route_summary(df)
    st.dataframe(route_summary.head(10).round(1), use_container_width=True)


# Route Analysis
# ===================================================

elif page == "Route Analysis":
    st.title("Route Analysis")

    if selected_origin == "All" or selected_dest == "All":
        st.info("Please select both a departure and arrival airport in the sidebar.")
    else:
        route_df = flights[
            (flights["origin"] == selected_origin) & (flights["dest"] == selected_dest)
        ]
        if len(route_df) == 0:
            st.warning("No flights found for this route.")
        else:
            st.header(f"Route: {selected_origin} -> {selected_dest}")

            fig_map = make_route_map(selected_origin, selected_dest, airports)
            if fig_map is not None:
                st.plotly_chart(fig_map, use_container_width=True)

            # numerical summaries using airport_delay_summary from Part 4
            summary = airport_delay_summary(route_df)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Flights on Route", f"{int(summary['n_flights']):,}")
            c2.metric("Avg Departure Delay", f"{summary['avg_dep_delay']:.1f} min")
            c3.metric("Avg Arrival Delay", f"{summary['avg_arr_delay']:.1f} min")
            c4.metric("On-Time Arrivals", f"{(route_df['on_time'].mean() * 100):.1f}%")

            st.divider()

            # delayed flights per quarter using amount_delayed_flights from Part 3
            st.subheader("Delayed Flights by Quarter")
            conn = sqlite3.connect("data/flights_database.db")
            q1 = amount_delayed_flights(conn, 1, 3, selected_dest)
            q2 = amount_delayed_flights(conn, 4, 6, selected_dest)
            q3 = amount_delayed_flights(conn, 7, 9, selected_dest)
            q4 = amount_delayed_flights(conn, 10, 12, selected_dest)
            conn.close()
            dq1, dq2, dq3, dq4 = st.columns(4)
            dq1.metric("Q1 (Jan–Mar)", f"{q1:,}")
            dq2.metric("Q2 (Apr–Jun)", f"{q2:,}")
            dq3.metric("Q3 (Jul–Sep)", f"{q3:,}")
            dq4.metric("Q4 (Oct–Dec)", f"{q4:,}")

            st.divider()

            # route statistics using airport_route_summary from Part 4
            st.subheader("Route Statistics")
            route_stats = airport_route_summary(route_df)
            st.dataframe(route_stats.round(1), use_container_width=True)

            # arrival delay by month for this route
            route_month = route_df.groupby("month")["arr_delay"].mean().reset_index()
            fig = px.line(route_month, x="month", y="arr_delay", markers=True,
                          title=f"Average Arrival Delay by Month ({selected_origin} → {selected_dest})",
                          labels={"month": "Month", "arr_delay": "Avg Arrival Delay (min)"})
            fig.update_xaxes(dtick=1)
            st.plotly_chart(fig, use_container_width=True)


# Delay Analysis
# ===================================================

elif page == "Delay Analysis":
    st.title("Delay Analysis")

    # departure delay distribution as seaborn histogram
    # clip to -60 to 120 min so the chart is readable (most flights fall in this range)
    st.subheader("Distribution of Departure Delays")
    hist_data = df[df["dep_delay"].between(-60, 120)]
    fig_hist, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(data=hist_data, x="dep_delay", bins=50, ax=ax)
    ax.set_xlabel("Departure Delay (min)")
    ax.set_ylabel("Number of Flights")
    ax.set_xlim(-60, 120)
    plt.tight_layout()
    st.pyplot(fig_hist)

    st.divider()

    # average departure delay by hour
    delay_by_hour = df.groupby("hour")["dep_delay"].mean().reset_index()
    fig = px.bar(delay_by_hour, x="hour", y="dep_delay",
                 title="Average Departure Delay by Hour of Day",
                 labels={"hour": "Hour", "dep_delay": "Average Delay (min)"})
    st.plotly_chart(fig, use_container_width=True)

    # wind speed vs departure delay
    wind_data = df.dropna(subset=["wind_speed", "dep_delay"])
    if len(wind_data) > 0:
        fig = px.scatter(
            wind_data.sample(min(3000, len(wind_data)), random_state=42),
            x="wind_speed", y="dep_delay",
            title="Wind Speed vs Departure Delay",
            labels={"wind_speed": "Wind Speed (mph)", "dep_delay": "Departure Delay (min)"})
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # correlation heatmap using seaborn
    st.subheader("Correlation Between Delay Factors")
    fig_corr = make_correlation_heatmap(df)
    if fig_corr is not None:
        st.pyplot(fig_corr)
    else:
        st.warning("Not enough data to compute correlations for the selected filters.")

    st.divider()

    # wind classification using compute_inner_product and classify_wind from Part 3
    st.subheader("Headwind vs Tailwind Effect on Delays")
    wind_df = df.dropna(subset=["wind_speed", "wind_dir", "dep_delay"]).copy()
    if len(wind_df) > 0:
        dest_coords = airports[["faa", "lat", "lon"]].copy()
        NYC_LAT, NYC_LON = 40.7128, -74.0060
        dest_coords["flight_direction"] = np.degrees(
            np.arctan2(
                np.radians(dest_coords["lon"] - NYC_LON),
                np.radians(dest_coords["lat"] - NYC_LAT)
            )
        ) % 360

        wind_df = wind_df.merge(
            dest_coords[["faa", "flight_direction"]],
            left_on="dest", right_on="faa", how="left"
        ).dropna(subset=["flight_direction"])

        wind_df["inner_product"] = compute_inner_product(
            wind_df["flight_direction"], wind_df["wind_dir"], wind_df["wind_speed"]
        )
        wind_df["wind_class"] = wind_df["inner_product"].apply(classify_wind)

        wind_delay = wind_df.groupby("wind_class")["dep_delay"].mean().reset_index()
        fig = px.bar(wind_delay, x="wind_class", y="dep_delay",
                     title="Average Departure Delay by Wind Type",
                     labels={"wind_class": "Wind Classification", "dep_delay": "Avg Delay (min)"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No wind data available for the selected filters.")

    st.divider()

    # most delayed destination airports
    delay_airports = (
        df.groupby("dest").agg(avg_arr=("arr_delay", "mean"), n=("flight", "count"))
        .reset_index()
    )
    delay_airports = delay_airports[delay_airports["n"] >= 50]
    delay_airports = delay_airports.sort_values("avg_arr", ascending=False).head(10)
    fig = px.bar(delay_airports, x="dest", y="avg_arr",
                 title="Most Delayed Destination Airports",
                 labels={"dest": "Airport", "avg_arr": "Avg Arrival Delay (min)"})
    st.plotly_chart(fig, use_container_width=True)


# Daily Statistics
# ===================================================

elif page == "Daily Statistics":
    st.title("Daily Statistics")

    # date picker only shown on this page
    min_date = flights["date"].min().date()
    max_date = flights["date"].max().date()
    selected_date = st.date_input("Select a date", value=min_date,
                                  min_value=min_date, max_value=max_date)

    st.header(f"Statistics for {selected_date.strftime('%Y-%m-%d')}")
    day_df = df[df["date"].dt.date == selected_date]

    if len(day_df) == 0:
        st.warning("No flights found for this date with the current filters.")
    else:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Flights That Day", f"{len(day_df):,}")
        d2.metric("Avg Departure Delay", f"{day_df['dep_delay'].mean():.1f} min")
        d3.metric("Median Departure Delay", f"{day_df['dep_delay'].median():.1f} min")
        d4.metric("On-Time Arrivals", f"{(day_df['on_time'].mean() * 100):.1f}%")

        # flight statistics using flight_statistics_for_day from Part 3
        if selected_origin != "All":
            st.subheader(f"Flight Statistics from {selected_origin}")
            conn = sqlite3.connect("data/flights_database.db")
            day_stats = flight_statistics_for_day(
                conn, selected_date.month, selected_date.day, selected_origin
            )
            conn.close()

            if day_stats and day_stats["n_flights"] > 0:
                s1, s2, s3 = st.columns(3)
                s1.metric("Most Visited Dest", f"{day_stats['most_visited_destination']} ({day_stats['most_visited_count']}x)")
                s2.metric("Furthest Flight", f"{day_stats['furthest_destination']} ({day_stats['furthest_distance']:.0f} miles)")
                s3.metric("Shortest Flight", f"{day_stats['shortest_destination']} ({day_stats['shortest_distance']:.0f} miles)")
        else:
            st.info("Select a departure airport in the sidebar for detailed daily statistics.")

        st.divider()

        # top 10 most delayed destinations on that day
        delay_dest = (
            day_df.groupby("dest")["arr_delay"].mean()
            .sort_values(ascending=False).head(10).reset_index()
        )
        fig = px.bar(delay_dest, x="dest", y="arr_delay",
                     title=f"Most Delayed Destinations on {selected_date.strftime('%Y-%m-%d')}",
                     labels={"dest": "Destination", "arr_delay": "Avg Arrival Delay (min)"})
        st.plotly_chart(fig, use_container_width=True)

        # airline summary table
        st.subheader("Airline Summary for This Day")
        day_table = (
            day_df.groupby("airline_name").agg(
                n=("flight", "count"), avg_dep=("dep_delay", "mean"),
                median_dep=("dep_delay", "median"), avg_arr=("arr_delay", "mean")
            ).reset_index().sort_values("n", ascending=False).round(1)
        )
        st.table(day_table)


# Fleet & Airlines
# ===================================================

elif page == "Fleet & Airlines":
    st.title("Fleet & Airlines")

    # airline performance summary
    airline_perf = (
        df.groupby("airline_name").agg(
            avg_arr=("arr_delay", "mean"), median_arr=("arr_delay", "median"),
            std_arr=("arr_delay", "std"), on_time_pct=("on_time", "mean"),
            n=("flight", "count")
        ).reset_index()
    )
    airline_perf = airline_perf[airline_perf["n"] >= 100]
    airline_perf["on_time_pct"] = airline_perf["on_time_pct"] * 100

    fig = px.bar(airline_perf.sort_values("avg_arr"), x="airline_name", y="avg_arr",
                 title="Average Arrival Delay per Airline (Best -> Worst)",
                 labels={"airline_name": "Airline", "avg_arr": "Avg Arrival Delay (min)"})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # performance table with median, std, sample size
    st.subheader("Airline Performance Summary")
    st.dataframe(airline_perf.sort_values("avg_arr").round(1), use_container_width=True)

    st.divider()

    # rainy vs clear day comparison
    rainy_perf = df.groupby(["airline_name", "rainy"])["arr_delay"].mean().reset_index()
    rainy_perf["weather"] = rainy_perf["rainy"].map({True: "Rainy", False: "Clear"})
    fig = px.bar(rainy_perf, x="airline_name", y="arr_delay", color="weather", barmode="group",
                 title="Airline Arrival Delay: Rainy vs Clear Days",
                 labels={"airline_name": "Airline", "arr_delay": "Avg Arrival Delay (min)"})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # top 5 manufacturers using top5_manufacturers from Part 3
    st.subheader("Top 5 Manufacturers by Destination")
    if selected_dest != "All":
        conn = sqlite3.connect("data/flights_database.db")
        top5 = top5_manufacturers(conn, selected_dest)
        conn.close()
        if len(top5) > 0:
            top5_df = top5.reset_index()
            top5_df.columns = ["Manufacturer", "Number of Flights"]
            fig = px.bar(top5_df, x="Manufacturer", y="Number of Flights",
                         title=f"Top 5 Manufacturers Flying to {selected_dest}")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    else:
        top_manu = df["manufacturer"].dropna().value_counts().head(10).reset_index()
        top_manu.columns = ["manufacturer", "n"]
        fig = px.bar(top_manu, x="manufacturer", y="n",
                     title="Top Manufacturers by Number of Flights",
                     labels={"manufacturer": "Manufacturer", "n": "Number of Flights"})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        st.info("Select a specific arrival airport to see the top 5 manufacturers for that destination.")


# Extra Insights
# ===================================================

elif page == "Extra Insights":
    st.title("Extra Insights")
    st.markdown("Additional analyses exploring delays, weather impact, and fleet characteristics.")

    # worst month for delays
    st.subheader("Worst Month for Delays")
    avg_delay_month = df.groupby("month")["arr_delay"].mean().reset_index()
    avg_delay_month.columns = ["month", "avg_arr_delay"]
    worst_month = avg_delay_month.loc[avg_delay_month["avg_arr_delay"].idxmax()]

    st.metric("Worst Month", f"Month {int(worst_month['month'])}", f"{worst_month['avg_arr_delay']:.2f} min avg delay")
    fig = px.bar(avg_delay_month, x="month", y="avg_arr_delay",
                 title="Average Arrival Delay per Month",
                 labels={"month": "Month", "avg_arr_delay": "Average Arrival Delay (min)"})
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # delays by origin airport
    st.subheader("Delays by Origin Airport")
    delay_by_origin = df.groupby("origin").agg(
        avg_dep_delay=("dep_delay", "mean"),
        avg_arr_delay=("arr_delay", "mean"),
        n_flights=("flight", "count")
    ).reset_index()
    fig = px.bar(delay_by_origin, x="origin", y=["avg_dep_delay", "avg_arr_delay"], barmode="group",
                 title="Average Delay by NYC Origin Airport",
                 labels={"value": "Average Delay (min)", "origin": "Airport"})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(delay_by_origin.sort_values("avg_dep_delay", ascending=False).round(1), use_container_width=True)

    st.divider()

    # airline weather resilience
    st.subheader("Airline Weather Resilience")
    weather_impact = df.groupby(["airline_name", "bad_weather"])["arr_delay"].mean().reset_index()
    weather_pivot = weather_impact.pivot(index="airline_name", columns="bad_weather", values="arr_delay")
    weather_pivot.columns = ["clear_delay", "bad_delay"]
    weather_pivot["delay_increase"] = weather_pivot["bad_delay"] - weather_pivot["clear_delay"]
    weather_pivot = weather_pivot.dropna().sort_values("delay_increase")

    fig = px.bar(weather_pivot.reset_index(), x="airline_name", y="delay_increase",
                 title="Extra Delay in Bad Weather per Airline (lower = more resilient)",
                 labels={"airline_name": "Airline", "delay_increase": "Extra Delay in Bad Weather (min)"})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # plane age vs weather delay
    st.subheader("Plane Age vs Weather Delay")
    age_df = df.dropna(subset=["plane_age"]).copy()
    if len(age_df) > 0:
        age_df["age_group"] = pd.cut(age_df["plane_age"], bins=[0, 5, 10, 15, 20, 50],
                                     labels=["0-5 yr", "5-10 yr", "10-15 yr", "15-20 yr", "20+ yr"])
        age_weather = age_df.dropna(subset=["age_group"]).groupby(
            ["age_group", "bad_weather"])["arr_delay"].mean().reset_index()
        age_weather["weather"] = age_weather["bad_weather"].map({True: "Bad Weather", False: "Clear"})

        fig = px.bar(age_weather, x="age_group", y="arr_delay", color="weather", barmode="group",
                     title="Plane Age vs Arrival Delay in Clear vs Bad Weather",
                     labels={"age_group": "Plane Age", "arr_delay": "Average Arrival Delay (min)"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No plane age data available for the selected filters.")

    st.divider()

    # plane size vs wind delay
    st.subheader("Plane Size vs Wind Delay")
    size_df = df.dropna(subset=["seats"]).copy()
    if len(size_df) > 0:
        size_df["size_group"] = pd.cut(size_df["seats"], bins=[0, 50, 150, 250, 500],
                                       labels=["Small (<50)", "Medium (50-150)", "Large (150-250)", "Very Large (>250)"])
        size_df["windy"] = size_df["wind_speed"] > 20
        size_wind = size_df.dropna(subset=["size_group"]).groupby(
            ["size_group", "windy"])["arr_delay"].mean().reset_index()
        size_wind["condition"] = size_wind["windy"].map({True: "Windy", False: "Calm"})

        fig = px.bar(size_wind, x="size_group", y="arr_delay", color="condition", barmode="group",
                     title="Plane Size vs Arrival Delay: Windy vs Calm Conditions",
                     labels={"size_group": "Plane Size", "arr_delay": "Avg Arrival Delay (min)"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No seat data available for the selected filters.")

    st.divider()

    # airline fleet age
    st.subheader("Airline Fleet Age")
    fleet_df = df.dropna(subset=["plane_age"])
    if len(fleet_df) > 0:
        avg_plane_age = fleet_df.groupby("airline_name").agg(
            avg_plane_age=("plane_age", "mean"),
            median_plane_age=("plane_age", "median"),
            n_flights=("flight", "count")
        ).reset_index().sort_values("avg_plane_age")

        fig = px.bar(avg_plane_age, x="airline_name", y="avg_plane_age",
                     title="Average Plane Age per Airline (lower = newer fleet)",
                     labels={"airline_name": "Airline", "avg_plane_age": "Avg Plane Age (years)"})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(avg_plane_age.round(1), use_container_width=True)
    else:
        st.warning("No plane age data available for the selected filters.")