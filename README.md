# Flights_Project_4
Data Engineering group project by Sharif el Gerf and Dalya el Gerf on monitoring flight information in 2023.

## Part 1
Part 1 uses 'airports.csv' for analysis and visualization. With the files for part 1:
- data/airports.csv
- scripts/part_1_functions.py
- scripts/main_part_1 (this can be used to run the part 1 figures and analysis)
### Part 1 uses pandas, numpy and plotly for the results

## Part 3
Part 3 uses "flights_database.db" to interact with the full database and link different tables together using SQL queries with sqlite3.
With the files for Part 3:
- data/flights_database.db
- scripts/part_3_functions.py
- scripts/main_part_3.py (this can be used to run the Part 3 figures and analysis)

### Part 3 uses sqlite3, pandas, numpy, matplotlib and plotly for the results

## Part 4
Part 4 uses "flights_database.db" for data wrangling, data cleaning, datetime conversion, consistency checks, and grouped summaries for later dashboard use.

With the files for Part 4:
- scripts/part_4_functions.py
- scripts/main_part_4.py (this can be used to run the Part 4 wrangling and analysis)

### Part 4 uses sqlite3, pandas, numpy and matplotlib for the results

## Part 5  
Part 5 implements an interactive dashboard using Streamlit to visualize NYC flight data. With the files for Part 5:  

- scripts/main_part_5.py (used to run the dashboard and visualizations)
- data/flights_database.db (used for full flight, airline, plane, and weather data)

The dashboard contains multiple sections for analysis:
- Overview: general statistics of flights departing from NYC, including numerical summaries and graphical visualizations.
- Route Analysis: selection of departure and arrival airports with route maps and route-specific metrics.
- Delay Analysis: distributions of departure delays, correlation with weather and wind effects, and identification of most delayed destinations.
- Daily Statistics: statistics for flights on a selected date, including average and median delays and airline summaries.
- Fleet & Airlines: performance metrics per airline, comparison of rainy vs clear days, and top manufacturers for destinations.

### Part 5 uses Streamlit, Plotly, Matplotlib, Seaborn, Pandas, NumPy, and SQLite for the results.  