import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import requests
import polyline

# Function to get road-following route using Google Directions API
def get_road_route(start, waypoints, end, api_key):
    if not api_key:
        st.error("Google Maps API key is missing.")
        return []
    
    url = f"https://maps.googleapis.com/maps/api/directions/json?origin={start}&destination={end}&waypoints={'|'.join(waypoints)}&mode=driving&key={api_key}"
    response = requests.get(url)
    data = response.json()
    
    if data["status"] == "OK":
        points = data["routes"][0]["overview_polyline"]["points"]
        return polyline.decode(points)
    else:
        st.error(f"Error fetching route: {data['status']}")
        return []

# Function to render map
def render_map(df, selected_vehicle, max_drive_time=600, api_key='AIzaSyDB-5cPWG__H3J38sloPutWcPLEgb1LYpM'):
    df = df.copy()
    df["coordinate"] = df["coordinate"].astype(str).fillna("")
    df["coordinate"] = df["coordinate"].str.replace(r"[()]", "", regex=True)
    df["Store ID"] = df["Store ID"].astype(str)
    df[["latitude", "longitude"]] = df["coordinate"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    
    vehicle_routes = df.groupby("Vehicle ID")[['latitude', 'longitude', 'Store ID', 'Travel Time to Next',"Layover_adjusted_arrival_time","Layover_adjusted_departure_time","Unloading Time","Window Start","Window End","New Wait Time","Distance to Next","Remaining Cube","Minutes_of_service","layover_route","layover_after"]].apply(lambda x: x.values.tolist())
    if not df.empty:
        first_lat, first_lon = df.iloc[0][["latitude", "longitude"]]
        m = folium.Map(location=[first_lat, first_lon], zoom_start=7)
    else:
        m = folium.Map(location=[40.02633, -79.0835], zoom_start=7)
    
    if selected_vehicle == "All":
        for vehicle, stops in vehicle_routes.items():
            add_vehicle_route(m, stops, max_drive_time, api_key)
    elif selected_vehicle in vehicle_routes:
        add_vehicle_route(m, vehicle_routes[selected_vehicle], max_drive_time, api_key)
    
    return m

# Function to add a vehicle's route to the map
def add_vehicle_route(m, stops, max_drive_time, api_key):
    total_drive_time = 0
    layover_stores = []
    
    if len(stops) > 1:
        start_location = f"{stops[0][0]},{stops[0][1]}"
        end_location = f"{stops[-1][0]+0.001},{stops[-1][1]+0.001}"
        waypoints = [f"{lat},{lon}" for lat, lon, *_ in stops[1:-1]]
        
        road_route = get_road_route(start_location, waypoints, end_location, api_key)
        if road_route:
            folium.PolyLine(road_route, color="blue", weight=3).add_to(m)
    
    stop_number = 0
    for lat, lon, store_id, travel_time,adj_arrival,adj_departure,unloading_time,start_time,end_time,wait_time,dist_next,remaining_cube,minutes_of_service,layover_route,layover_after in stops:
        total_drive_time += travel_time
        if total_drive_time > max_drive_time:
            # popup_text = (
            #     f"<b>Stop {stop_number}</b><br>"
            #     f"Store ID: {store_id}<br>"
            #     f"Adjusted Arrival: {adj_arrival}<br>"
            #     f"Adjusted Departure: {adj_departure}<br>"
            #     f"Start Time: {start_time}<br>"
            #     f"End Time: {end_time}<br>"
            #     f"Unloading Time: {unloading_time}<br>"
            #     f"Wait Time: {wait_time} min<br>"
            #     f"Distance to Next: {dist_next} miles<br>"
            #     f"Travel Time: {travel_time} min<br>"
            #     f"Remaining Cube: {remaining_cube}<br>"
            #     f"Minutes Of Service: {minutes_of_service}"
            # )
            # folium.Marker([lat, lon], popup=popup_text, icon=folium.Icon(color="red", icon="pause"))
            popup_text = f"""
            <div style="width: 150px; white-space: normal;">
                <b>Stop {stop_number}</b><br>
                Store ID: {store_id}<br>
                Arrival Time: {adj_arrival}<br>
                Departure Time: {adj_departure}<br>
                Start Time: {start_time}<br>
                End Time: {end_time}<br>
                Unloading Time: {unloading_time}<br>
                Wait Time: {wait_time} min<br>
                Distance to Next: {dist_next} miles<br>
                Travel Time: {travel_time} min<br>
                Remaining Cube: {remaining_cube}<br>
                Minutes Of Service: {minutes_of_service}<br>
                layover_route: {layover_route}<br>
                layover_after: {layover_after}
            </div>
        """
            folium.Marker(
                [lat, lon], 
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color="blue")
            ).add_to(m)
            layover_stores.append(store_id)
            total_drive_time = 0  # Reset after layover
        else:
            # popup_text = (    
            #     f"<b>Stop {stop_number}</b><br>"
            #     f"Store ID: {store_id}<br>"
            #     f"Arrival Time: {adj_arrival}<br>"
            #     f"Departure Time: {adj_departure}<br>"
            #     f"Start Time: {start_time}<br>"
            #     f"End Time: {end_time}<br>"
            #     f"Unloading Time: {unloading_time}<br>"
            #     f"Wait Time: {wait_time} min<br>"
            #     f"Distance to Next: {dist_next} miles<br>"
            #     f"Travel Time: {travel_time} min<br>"
            #     f"Remaining Cube: {remaining_cube}<br>"
            #     f"Minutes Of Service: {minutes_of_service}"
            # )
            popup_text = f"""
            <div style="width: 150px; white-space: normal;">
                <b>Stop {stop_number}</b><br>
                Store ID: {store_id}<br>
                Arrival Time: {adj_arrival}<br>
                Departure Time: {adj_departure}<br>
                Start Time: {start_time}<br>
                End Time: {end_time}<br>
                Unloading Time: {unloading_time}<br>
                Wait Time: {wait_time} min<br>
                Distance to Next: {dist_next} miles<br>
                Travel Time: {travel_time} min<br>
                Remaining Cube: {remaining_cube}<br>
                Minutes Of Service: {minutes_of_service}<br>
                layover_route: {layover_route}<br>
                layover_after: {layover_after}
            </div>
        """
            folium.Marker(
                [lat, lon], 
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color="blue")
            ).add_to(m)

            # folium.Marker([lat, lon], popup=popup_text, icon=folium.Icon(color="blue")).add_to(m)
        stop_number += 1
    
    return layover_stores

# Streamlit UI
st.title("Vehicle Route Optimization with Layover Highlighting")
