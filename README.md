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
- Extra Insights: additional analyses exploring worst month for delays, delays by origin airport, airline weather resilience, plane age vs weather, plane size vs wind conditions, and airline fleet age.

### Part 5 uses Streamlit, Plotly, Matplotlib, Seaborn, Pandas, NumPy, and SQLite for the results

## Extra Analysis
We also include an optional script for additional exploratory analysis outside the dashboard. With the files:
- scripts/run_extra.py (standalone script performing extra calculations and visualizations similar to Part 5 Extra Insights)

### This script uses sqlite3, pandas, numpy, plotly, and datetime for the results