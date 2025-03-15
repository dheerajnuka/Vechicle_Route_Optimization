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

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data", df.head())

    # Process routes
    full_final_df,vehicle_summary = process_routes(df)
    st.write("### Routes Data", full_final_df)
    st.write("### Vehicle Summary", vehicle_summary)

    # Render map
    st.write("### Route Map")
    vehicle_ids = full_final_df["Vehicle ID"].unique().tolist()
    vehicle_ids.insert(0, "All")
    selected_vehicle = st.selectbox("Select Vehicle ID", vehicle_ids)
    route_map =render_map(full_final_df,selected_vehicle)
    folium_static(route_map)