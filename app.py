import streamlit as st
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils.process_routes import process_routes
from components.map_renderer import render_map
from streamlit_folium import folium_static

st.title("📍 Route Mapping Application")
st.write("Upload a CSV file to visualize routes.")
# hide_github_icon = """
# #GithubIcon {
#   visibility: hidden;
# }
# """
# st.markdown(hide_github_icon, unsafe_allow_html=True)
# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data", df.head())

    # Process routes
    full_final_df,vehicle_summary = process_routes(df)
    cols_to_drop = [
        "Location",
        "Cube",
        "layover_after",
        "layover_before",
        "Layover_adjusted_arrival_time",
        "Layover_adjusted_departure_time",
        "original_wait_time",
        "storeID"
    ]

    # Drop them (ignore if a column doesn’t exist)
    full_final_df1 = full_final_df.drop(columns=cols_to_drop, errors="ignore")
    cols_to_drop1= [
        "End Time in min",
        "Start Time in min",
        "capacity",
        "new_capacity"
    ]
    vehicle_summary1= vehicle_summary.drop(columns=cols_to_drop1,errors='ignore')
    st.write("### Routes Data", full_final_df1)
    st.write("### Vehicle Summary", vehicle_summary1)

    # Render map
    st.write("### Route Map")
    vehicle_ids = full_final_df["Vehicle ID"].unique().tolist()
    vehicle_ids.insert(0, "All")
    selected_vehicle = st.selectbox("Select Vehicle ID", vehicle_ids)
    route_map =render_map(full_final_df,selected_vehicle)
    folium_static(route_map)