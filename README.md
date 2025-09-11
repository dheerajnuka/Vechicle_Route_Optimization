# Streamlit Route Mapping App 🚀

## 📌 Features
- Upload CSV file
- Process routes with a mock function
- Display routes on a map using Folium

## 🛠 Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```


## Data Dictionary — Vehicle Route Optimization

This table documents each column in the route output.

| Column name | Description |
|---|---|
| Vehicle ID | Unique identifier of the vehicle running the route/stop. |
| Store ID | Unique identifier of the delivery/pickup location (customer/site). |
| Cube | Volume to deliver/pick up at this stop (e.g., cubic ft/m). |
| Remaining Cube | Vehicle’s remaining usable volume **after** servicing this stop. |
| Arrival Time | Planned time the vehicle reaches the stop (before any waiting). |
| departure_time | Time the vehicle leaves the stop after service is completed. |
| Suggested Store Open Time | Earliest recommended store opening time used by planning rules. |
| Adjusted Arrival Time | Arrival shifted to respect opening hours/time-windows/breaks. |
| Minutes_of_service | Total service duration at the stop (unload, paperwork, scan, etc.). |
| layover_route | Flag/identifier for a planned layover/break inserted in the route. |
| New Wait Time | Recomputed wait after any arrival adjustments/layovers. |
| Window Start | Earliest allowed service start time at the stop (time window). |
| Window End | Latest allowed service start/finish boundary at the stop (time window). |
| Distance to Next | Road distance from this stop to the next planned stop. |
| Travel Time to Next | Estimated driving time to the next stop. |
| Wait Time | Time spent waiting if arriving before the service window/open time. |
| Unloading Time | Portion of service time specifically for unloading goods. |
| Validation | Result/status of constraint checks (on-time, capacity, length, etc.). |
| Max Trailer Length (Store) | Longest trailer the location can accommodate. |
| Vehicle Trailer Length | Trailer length of the assigned vehicle. |
| Store Preferred Vehicle Type | Preferred truck/class/features for the location (e.g., reefer, liftgate). |
| Vehicle Type | Actual vehicle class/type used for this stop/route. |
| Dispatch Date | Operating date for the route/stop plan. |
| Coordinate | Latitude/longitude of the store/location. |

**Typical relationships**
- `AdjustedArrival = max(Arrival Time, Window Start, Suggested Store Open Time)`
- `Wait Time = max(0, Window Start − Arrival Time)`
- `New Wait Time = max(0, Window Start − Adjusted Arrival Time)`
- `departure_time = Adjusted Arrival Time + Minutes_of_service`
- `Remaining Cube(next) = Remaining Cube(current) − Cube` (deliveries)