import streamlit as st
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils.process_routes import process_routes
from components.map_renderer import render_map
from streamlit_folium import folium_static

st.title("📍 Route Mapping Application")

# --- README link ---
README_URL = "https://github.com/dheerajnuka/Vechicle_Route_Optimization/blob/main/README.md"
st.caption("📘 Need help? Read the project README for output route,vehicle summary columns and sample CSV file")
if hasattr(st, "link_button"):  # Streamlit >= 1.25
    st.link_button("Open README", README_URL)
else:
    st.markdown(
        f'<a href="{README_URL}" target="_blank" rel="noopener noreferrer" '
        'style="display:inline-block;padding:8px 14px;border:1px solid #e6e6e6;'
        'border-radius:8px;text-decoration:none;">📘 Open README</a>',
        unsafe_allow_html=True
    )
# --- end README link ---

st.write("Upload a CSV file to visualize routes.")
# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data", df.head())

    # Process routes
    full_final_df, vehicle_summary = process_routes(df)
    cols_to_drop = [
        "Location",
        "layover_after",
        "layover_before",
        "Layover_adjusted_arrival_time",
        "Layover_adjusted_departure_time",
        "original_wait_time",
        "storeID"
    ]

    # Drop them (ignore if a column doesn’t exist)
    full_final_df1 = full_final_df.drop(columns=cols_to_drop, errors="ignore")
    full_final_df2 = full_final_df1[[
        "Vehicle ID","Store ID","Cube","Remaining Cube","Arrival Time","departure_time",
        "Distance to Next","Travel Time to Next","Unloading Time","Suggested Store Open Time",
        "Adjusted Arrival Time","Minutes_of_service","layover_route","New Wait Time",
        "Window Start","Window End","Validation","Max Trailer Length (Store)",
        "Vehicle Trailer Length","Store Preferred Vehicle Type","Vehicle Type",
        "Dispatch Date","Coordinate"
    ]]
    cols_to_drop1 = ["End Time in min", "Start Time in min"]
    vehicle_summary1 = vehicle_summary.drop(columns=cols_to_drop1, errors='ignore')

    st.write("### Routes Data", full_final_df2)
    st.write("### Vehicle Summary", vehicle_summary1)

    # Render map
    st.write("### Route Map")
    vehicle_ids = full_final_df["Vehicle ID"].unique().tolist()
    vehicle_ids.insert(0, "All")
    selected_vehicle = st.selectbox("Select Vehicle ID", vehicle_ids)
    route_map = render_map(full_final_df, selected_vehicle)
    folium_static(route_map)
