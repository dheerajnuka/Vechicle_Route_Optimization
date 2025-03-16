import pandas as pd
import re
from geopy.distance import geodesic
import itertools
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from datetime import timedelta
import numpy as np
import os
def time_to_minutes(time_str):
    # print(time_str)
    hours, minutes = map(int, time_str.split(":"))
    return hours * 60 + minutes


def minutes_to_24hr_time_string(minutes):
    """Converts minutes into a 24-hour HH:MM format string."""
    time = timedelta(minutes=minutes)
    total_seconds = time.total_seconds()
    hours = int(total_seconds // 3600) % 24  # Ensures 24-hour format
    minutes = int((total_seconds % 3600) // 60)
    return f"{hours:02}:{minutes:02}"

def create_data_model_from_excel(excel_file="file_input.xlsx", num_vehicles=10, depot=0):
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
    excel_file = os.path.join(base_dir, excel_file)  # Construct full path
    data = {}
    distance_matrix = pd.read_excel(excel_file, sheet_name='DistanceMatrix', index_col=0).values.astype(int).tolist()
    time_matrix = pd.read_excel(excel_file, sheet_name='TimeMatrix', index_col=0).values.tolist()
    time_windows_df = pd.read_excel(excel_file, sheet_name='TimeWindows')
    store_demands_df =pd.read_excel(excel_file, sheet_name='StoreDemands')
    store_ids = time_windows_df['location'].tolist()  
    vehicle_capacities = pd.read_excel(excel_file, sheet_name='VehicleCapacities', index_col=0)['new_capacity'].values.tolist()
    unloading_times =pd.read_excel(excel_file, sheet_name='StoreDemands', index_col=0)['unloading_time'].values.tolist()
    store_vehicle_types = pd.read_excel(excel_file, sheet_name='TimeWindows', index_col=0)['lift_gate_types'].values.tolist()
    vehicle_types = pd.read_excel(excel_file, sheet_name='VehicleCapacities', index_col=0)['vehicle_types'].values.tolist()

    time_windows = []
    for _, row in time_windows_df.iterrows():
        start = time_to_minutes(row['start_time'])
        end = time_to_minutes(row['end_time'])
        if start <= end:
            time_windows.append((start, end))
        else:
            time_windows.append((start, 2879))
            time_windows.append((0, end))
    store_max_trailer_lengths=time_windows_df['store_max_trailer_lengths'].tolist()
    vehicle_trailer_lengths = pd.read_excel(excel_file, sheet_name='VehicleCapacities', index_col=0)['vehicle_trailer_lengths'].values.tolist()
    demands = store_demands_df['demand'].astype(int).tolist()
    # Add unloading time to the data from StoreDemands
    # unloading_time_df = time_windows_df.copy()
    # unloading_time_df['unloading_time'] = unloading_time_df['unloading_time']
    # unloading_times = unloading_time_df['unloading_time'].tolist()
    depot_start_time, depot_end_time = -1440, 2879
    time_windows[depot] = (depot_start_time, depot_end_time)
    data["distance_matrix"] = distance_matrix
    data["time_matrix"] = time_matrix
    data["num_vehicles"] = num_vehicles
    data["depot"] = depot
    data["demands"] = demands
    data["vehicle_capacities"] = vehicle_capacities
    data["time_windows"] = time_windows
    data["store_ids"] = store_ids
    data["unloading_times"] = unloading_times
    data['store_max_trailer_lengths']=store_max_trailer_lengths
    data['vehicle_trailer_lengths']=vehicle_trailer_lengths
    data['store_pref_vehicle_types']=store_vehicle_types
    data['vehicle_types']=vehicle_types
    return data

def solve_routing_problem_with_unloading(data):
    """
    Solves the vehicle routing problem with dynamic unloading time for each store,
    reduces wait time, ensures driver break time after 10 hours of work,
    and includes mandatory break callbacks.
    """
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(len(data["distance_matrix"]), data["num_vehicles"], data["depot"])
    
    # Create the routing model
    routing = pywrapcp.RoutingModel(manager)
    
    # Define the distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    # Register the distance callback
    distance_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)

    # Define the time callback
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel_time = data["time_matrix"][from_node][to_node]
        unloading_time = data["unloading_times"][from_node]
        total_time = travel_time + unloading_time
        return total_time

    # Register the time callback
    time_callback_index = routing.RegisterTransitCallback(time_callback)

    # Add the time dimension
    time_dimension_name = "Time"
    routing.AddDimension(
        time_callback_index,  # Callback index
        300,                  # Slack time
        14400,              # Maximum time (e.g., 10 hours + break time)
        False,              # Allow flexible start time
        time_dimension_name
    )
    
    time_dimension = routing.GetDimensionOrDie(time_dimension_name)

    depot_index = manager.NodeToIndex(data["depot"])
    time_dimension.SlackVar(depot_index).SetValue(0)

    # Add time windows for each location
    for location_idx, (start, end) in enumerate(data["time_windows"]):
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(start, end)

    # Add capacity constraints
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,  # Demand callback
        0,                      # Null capacity slack
        data["vehicle_capacities"],  # Vehicle capacities
        True,                   # Start cumul to zero
        "Capacity"
    )

    for vehicle_id in range(data["num_vehicles"]):
        start_index = routing.Start(vehicle_id)  # Get the start index for each vehicle
        time_dimension.SlackVar(start_index).SetValue(0)

    # Add penalties for missed locations
    penalty = 1000000000
    for location_idx in range(len(data["time_windows"])):
        index = manager.NodeToIndex(location_idx)
        routing.AddDisjunction([index], penalty)

    # Add trailer length constraints
    for store_idx in range(len(data["store_max_trailer_lengths"])):
        for vehicle_idx in range(data["num_vehicles"]):
            vehicle_length = data["vehicle_trailer_lengths"][vehicle_idx]
            max_length = data["store_max_trailer_lengths"][store_idx]
            if vehicle_length > max_length:
                store_index = manager.NodeToIndex(store_idx)
                routing.VehicleVar(store_index).RemoveValue(vehicle_idx)

    # Add vehicle type constraints
    for store_idx in range(len(data["store_pref_vehicle_types"])):
        for vehicle_idx in range(data["num_vehicles"]):
            vehicle_type = data["vehicle_types"][vehicle_idx]
            pref_types = data["store_pref_vehicle_types"][store_idx]
            if vehicle_type not in pref_types:
                store_index = manager.NodeToIndex(store_idx)
                routing.VehicleVar(store_index).RemoveValue(vehicle_idx)

    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 60

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    return data, solution, routing, manager, time_dimension


def minutes_to_24hr_time_string(minutes):
    hours = minutes // 60
    mins = minutes % 60
    return f"{int(hours):02d}:{int(mins):02d}"

def save_solution_to_excel(data, manager, routing, solution, time_dimension, output_file, 
                           break_after_minutes=10000, break_duration_minutes=5, depot_id=15501, dispatch_date=None):
    dispatch_date = '2024-08-26'
    # Initialize a list to store rows of data for the output
    rows = []

    # Iterate through each vehicle to process the assigned route
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)  # Get the starting index for the vehicle
        
        # Check if the vehicle has no assigned route
        if routing.IsEnd(index):
            print(f"Vehicle {vehicle_id} has no route assigned.")
            continue

        assigned_demand = 0  # Initialize total demand assigned to this vehicle
        # Collect all nodes (locations) assigned to this vehicle
        route_nodes = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)  # Convert internal index to node index
            route_nodes.append(node)
            assigned_demand += data["demands"][node]  # Accumulate demand
            index = solution.Value(routing.NextVar(index))  # Move to the next index

        remaining_load = assigned_demand  # Set the remaining load initially as the total demand

        # Reset the index to traverse the route again and log route details
        index = routing.Start(vehicle_id)
        total_travel_time = 0  # Track total travel time for break enforcement
        cumulative_arrival_time = solution.Min(time_dimension.CumulVar(index))  # Arrival time at the depot

        while not routing.IsEnd(index):
            previous_index = index
            index = solution.Value(routing.NextVar(index))

            # Retrieve information about travel between locations
            from_node = manager.IndexToNode(previous_index)
            to_node = manager.IndexToNode(index)
            distance_between = data["distance_matrix"][from_node][to_node]
            time_between = data["time_matrix"][from_node][to_node]

            # Unload the demand at the current location (store)
            store_demand = data["demands"][from_node]
            remaining_load -= store_demand  # Update the remaining load

            # Map the current location to a store ID or label it as "DC"
            store_id = data["store_ids"][from_node] if from_node < len(data["store_ids"]) else "DC"

            # Retrieve unloading time from the data (list-based access by store index)
            unloading_time_minutes = data["unloading_times"][from_node]  # Unloading time is indexed by node

            # Retrieve time window constraints for the current store
            time_window = data["time_windows"][from_node]
            window_start, window_end = time_window
            arrival_time_str = minutes_to_24hr_time_string(cumulative_arrival_time)
            window_start_str = minutes_to_24hr_time_string(window_start)
            window_end_str = minutes_to_24hr_time_string(window_end)

            # Calculate wait time if the vehicle arrives before the store opens
            wait_time = 0
            if cumulative_arrival_time < window_start:
                wait_time = window_start - cumulative_arrival_time  # Calculate wait time
                cumulative_arrival_time = window_start  # Adjust to the opening time
                arrival_time_str = minutes_to_24hr_time_string(cumulative_arrival_time)

            within_window = window_start <= cumulative_arrival_time <= window_end
            validation_message = "Valid" if within_window else "Invalid"

            # Print unloading time and add it to cumulative arrival time
            cumulative_arrival_time += unloading_time_minutes
           
            # Check if a break is needed based on accumulated travel time
            break_info, break_start_str, break_end_str = "", "", ""
            if total_travel_time >= break_after_minutes:
                # Schedule a break
                break_start = cumulative_arrival_time
                break_end = break_start + break_duration_minutes
                break_info = f"Break for {break_duration_minutes} minutes"
                break_start_str = minutes_to_24hr_time_string(break_start)
                break_end_str = minutes_to_24hr_time_string(break_end)

                # Update arrival time to account for the break
                cumulative_arrival_time = break_end
                total_travel_time = 0  # Reset the travel time after the break

            # Update arrival time with the time to the next location
            cumulative_arrival_time += time_between
            total_travel_time += time_between  # Accumulate travel time for break enforcement

            # Add trailer length details
            max_trailer_length = data["store_max_trailer_lengths"][from_node] if from_node < len(data["store_max_trailer_lengths"]) else np.nan
            vehicle_trailer_length = data["vehicle_trailer_lengths"][vehicle_id]

            # Retrieve store preferred vehicle type
            store_pref_vehicle_types = data["store_pref_vehicle_types"][from_node] if from_node < len(data["store_pref_vehicle_types"]) else np.nan
            vehicle_type = data["vehicle_types"][vehicle_id]

            # Append route details to the list of rows
            rows.append([vehicle_id, store_id, from_node, store_demand, remaining_load, arrival_time_str, window_start_str,
                         window_end_str, distance_between, time_between, wait_time, unloading_time_minutes,
                         validation_message, max_trailer_length, vehicle_trailer_length, store_pref_vehicle_types, vehicle_type])

        # **Add the final entry for returning to the depot (DC)**
        arrival_time_str = minutes_to_24hr_time_string(cumulative_arrival_time)
        
        # Retrieve time window constraints for the depot (DC)
        dc_time_window = data["time_windows"][0]
        dc_window_start, dc_window_end = dc_time_window
        dc_window_start_str = minutes_to_24hr_time_string(dc_window_start)
        dc_window_end_str = minutes_to_24hr_time_string(dc_window_end)

        rows.append([vehicle_id, "DC", manager.IndexToNode(index), 0, remaining_load, arrival_time_str, 
                     dc_window_start_str, dc_window_end_str, 0, 0, 0, 0, "Valid",
                     100.0, data["vehicle_trailer_lengths"][vehicle_id], "['reefer liftgate']", data["vehicle_types"][vehicle_id]])

    # Convert the list of rows into a DataFrame with the appropriate columns
    columns = ["Vehicle ID", "Store ID", "Location", "Cube", "Remaining Cube", "Arrival Time", "Window Start", 
               "Window End", "Distance to Next", "Travel Time to Next", "Wait Time", "Unloading Time", "Validation",
               "Max Trailer Length (Store)", "Vehicle Trailer Length", "Store Preferred Vehicle Type", "Vehicle Type"]
    output_df = pd.DataFrame(rows, columns=columns)

    # Add dispatch date to the DataFrame
    output_df["Dispatch Date"] = dispatch_date

    # Filter out routes with fewer than 3 locations (to exclude incomplete routes)
    output_df = output_df.groupby("Vehicle ID").filter(lambda x: len(x) > 2)
    return output_df

def add_suggested_store_open_time(df):
    """
    Add a column "Suggested Store Open Time" where "Wait Time" is greater than zero.
    This column suggests an earlier store opening time based on the logic:
    - Previous store suggested open time (if available) + unloading time + travel time to next store.
    - For consecutive stops with wait times, the suggested open time for the next stop should
      consider the previously calculated suggested open time.

    Args:
        df (pd.DataFrame): DataFrame generated from the previous code.

    Returns:
        pd.DataFrame: Updated DataFrame with a new column for suggested store opening times.
    """
    # Make a copy to avoid modifying the original DataFrame
    updated_df = df.copy()

    # Initialize the "Suggested Store Open Time" column with blank values
    updated_df["Suggested Store Open Time"] = ""

    # Iterate through the DataFrame by vehicle ID to ensure continuity of routes
    for vehicle_id, group in updated_df.groupby("Vehicle ID"):
        # Iterate directly on `updated_df` to ensure updates are reflected
        for i in range(1, len(group)):
            # Calculate suggested open time if wait time is greater than zero
            current_index = group.index[i]
            previous_index = group.index[i - 1]

            if updated_df.loc[current_index, "Wait Time"] > 0:
                # Check if the previous stop has a suggested open time
                prev_suggested_open_time = updated_df.loc[previous_index, "Suggested Store Open Time"]
                if prev_suggested_open_time:
                    prev_arrival_minutes = int(prev_suggested_open_time.split(":")[0]) * 60 + int(prev_suggested_open_time.split(":")[1])
                else:
                    # Fall back to the actual arrival time
                    prev_arrival_time = updated_df.loc[previous_index, "Arrival Time"]
                    prev_arrival_minutes = int(prev_arrival_time.split(":")[0]) * 60 + int(prev_arrival_time.split(":")[1])

                prev_unloading_time = updated_df.loc[previous_index, "Unloading Time"]
                travel_time_to_next = updated_df.loc[previous_index, "Travel Time to Next"]

                # Calculate the suggested open time in minutes
                suggested_open_time_minutes = prev_arrival_minutes + prev_unloading_time + travel_time_to_next

                # Convert the suggested open time back to HH:MM format
                suggested_open_time = f"{suggested_open_time_minutes // 60:02d}:{suggested_open_time_minutes % 60:02d}"

                # Update the current row with the calculated suggested open time
                updated_df.loc[current_index, "Suggested Store Open Time"] = suggested_open_time

    return updated_df

def adjust_arrival_times(updated_df):
    # Re-importing required libraries after execution state reset
    df=updated_df.copy()
    df=df.reset_index(drop=True)
    df['original_wait_time']=df['Wait Time']
    def time_to_minutes(time_str):
        if ":" in time_str:
            parts = time_str.split(":")
            return int(parts[0]) * 60 + int(parts[1])
        return float('-inf') if time_str.startswith('-') else float('inf')

    df["Arrival Time (mins)"] = df["Arrival Time"].apply(time_to_minutes)
    df["Window Start (mins)"] = df["Window Start"].apply(time_to_minutes)
    df["Window End (mins)"] = df["Window End"].apply(time_to_minutes)

    # Adjusting arrival times by reducing wait time while ensuring within time windows
    for i in range(len(df) - 1, 0, -1):  # Iterate in reverse to adjust backwards
        if df.loc[i, "Wait Time"] > 0:
            j = i - 1
            while j >= 0 and df.loc[j, "Vehicle ID"] == df.loc[i, "Vehicle ID"]:
                new_arrival_time = df.loc[j, "Arrival Time (mins)"] + df.loc[i, "Wait Time"]
                
                # Ensure arrival time is within store's time window
                if new_arrival_time > df.loc[j, "Window End (mins)"]:
                    possible_reduction = new_arrival_time - df.loc[j, "Window End (mins)"]
                    df.loc[i, "Wait Time"] -= min(df.loc[i, "Wait Time"], possible_reduction)
                    new_arrival_time = df.loc[j, "Arrival Time (mins)"] + df.loc[i, "Wait Time"]

                if new_arrival_time >= df.loc[j, "Window Start (mins)"]:
                    df.loc[j, "Arrival Time (mins)"] = new_arrival_time
                
                j -= 1  # Move backwards

    df["Adjusted Arrival Time"] = df["Arrival Time (mins)"].apply(minutes_to_time)

    # Recalculating new wait time after adjusted arrival times
    df["New Wait Time"] = 0  # Initialize column

    for i in range(len(df) - 1):
        if df.loc[i + 1, "Vehicle ID"] == df.loc[i, "Vehicle ID"]:  # Ensure within the same vehicle route
            df.loc[i+1, "New Wait Time"] = max(0, 
                df.loc[i + 1, "Arrival Time (mins)"] - df.loc[i, "Arrival Time (mins)"] - df.loc[i, "Travel Time to Next"] - df.loc[i, "Unloading Time"]
            )

    return df

def classify_vehicle_type(arrival_times):
        am = any(time < 720 for time in arrival_times if not np.isnan(time))  # Before 12:00 PM
        pm = any(time >= 720 for time in arrival_times if not np.isnan(time))  # 12:00 PM or later
        if am and pm:
            return "Both AM and PM"
        elif am:
            return "AM vehicle"
        else:
            return "PM vehicle"
def time_to_minutes(time_str):
    return int(time_str.split(":")[0]) * 60 + int(time_str.split(":")[1])

def vehicle_summary_calculation(updated_df,VehicleCapacities):
    # Ensure "Distance to Next" column is numeric, replacing non-numeric values with 0
    import warnings
    warnings.filterwarnings('ignore')
    updated_df["Distance to Next"] = pd.to_numeric(updated_df["Distance to Next"], errors='coerce').fillna(0).astype(int)

    # Convert "Arrival Time" to datetime in minutes, ignoring empty strings
    updated_df["Arrival Time (minutes)"] = updated_df["Arrival Time"].apply(
        lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]) if x else np.nan
    )
    vehicle_summary = updated_df.groupby("Vehicle ID",as_index=False).apply(lambda group: pd.Series({
        "Total Stores": (group["Store ID"] != "DC").sum(),
        "Start Time": group.loc[group["Store ID"] == "DC", "Arrival Time"].iloc[0],  # First depot arrival time
        "End Time": group.loc[group["Store ID"] == "DC", "Arrival Time"].iloc[-1],  # Last depot arrival time
        "Total Distance": group["Distance to Next"].sum(),
        "Total Load": group["Cube"].sum(),  # Sum of demands handled by the vehicle
        "Total Wait Time": group["Wait Time"].sum(),  # Sum of wait times for the vehicle
        "Vehicle Type": classify_vehicle_type(group["Arrival Time (minutes)"]),
    }))
    # Reset index to prepare for merging
    
    vehicle_summary = vehicle_summary.reset_index(drop=True)
    # Convert Start and End Times to minutes
    # print(vehicle_summary)
    # vehicle_summary.to_csv('vehicle_summary.csv',index=False)


    vehicle_summary['End Time in min'] = vehicle_summary['End Time'].apply(time_to_minutes)
    vehicle_summary['Start Time in min'] = vehicle_summary['Start Time'].apply(time_to_minutes)

    # Add "Time Travel" column in hours
    vehicle_summary["Time Travel (hours)"] = (vehicle_summary['End Time in min'] - vehicle_summary['Start Time in min']) / 60

    # Merge with vehicle capacities
    vehicle_capacities = VehicleCapacities[['vehicle_id', 'capacity','new_capacity', 'vehicle_types']]
    vehicle_capacities['vehicle_id'] = vehicle_capacities['vehicle_id'].astype('int64')
    vehicle_summary = vehicle_summary.merge(vehicle_capacities, left_on='Vehicle ID', right_on="vehicle_id", how="left")

    # Calculate trailer utilization (Cube/Capacity)
    vehicle_summary["Trailer Utilization (%)"] = (vehicle_summary["Total Load"] / vehicle_summary["capacity"]) * 100
    vehicle_summary["Haul Type"] = vehicle_summary["Time Travel (hours)"].apply(lambda x: "Long Haul" if x >= 10 else "Short Haul")
    # Clean up the merged dataframe
    vehicle_summary.drop(columns=['vehicle_id'], inplace=True)
    return vehicle_summary

def minutes_to_time(minutes):
    return f"{minutes // 60:02d}:{minutes % 60:02d}"

def layover_logics(df):
    df["Adjusted Arrival Time (mins)"] = df["Adjusted Arrival Time"].apply(time_to_minutes)
    df['departure_time'] = (df["Adjusted Arrival Time (mins)"] + df['Unloading Time']).apply(minutes_to_time)

    df["Unloading Time"] = df["Unloading Time"].astype(int)

    # Initialize the "Minutes_of_service" column with 0
    df["Minutes_of_service"] = 0

    # Group by Vehicle ID and apply the running sum logic within each vehicle's route
    for vehicle_id, group in df.groupby("Vehicle ID"):
        indices = group.index.tolist()
        df.at[indices[0], "Minutes_of_service"] = 0  # First row for each vehicle starts at 0

        for i in range(1, len(indices)):
            prev_index = indices[i - 1]
            curr_index = indices[i]

            df.at[curr_index, "Minutes_of_service"] = (
                df.at[prev_index, "Travel Time to Next"] +
                df.at[prev_index, "Unloading Time"] +
                df.at[prev_index, "Minutes_of_service"] +
                df.at[curr_index, "New Wait Time"]
            )

    df["layover_route"] = "No Layover"

    for vehicle_id, group in df.groupby("Vehicle ID"):
        last_index = group.index[-1]
        if df.at[last_index, "Minutes_of_service"] > 600:
            df.loc[group.index, "layover_route"] = "Layover"

    # Reset the layover columns to 0
    df["layover_after"] = 0
    df["layover_before"] = 0

    # Apply the layover conditions for each vehicle
    for vehicle_id, group in df.groupby("Vehicle ID"):
        indices = group.index.tolist()

        layover_triggered = False  # Flag to track when layover happens

        for i in range(1, len(indices)):  # Start from the second row to access the previous row
            curr_index = indices[i]
            prev_index = indices[i - 1]

            # If the current row has Minutes_of_service > 600, update layover columns
            if df.at[curr_index, "Minutes_of_service"] > 600 and not layover_triggered:
                df.at[prev_index, "layover_after"] = 1  # Mark previous row as layover_after
                df.at[curr_index, "layover_before"] = 1  # Mark current row as layover_before
                layover_triggered = True  # Ensure only one instance is marked

            # After marking one layover instance, reset all subsequent rows to 0
            elif layover_triggered:
                df.at[curr_index, "layover_after"] = 0
                df.at[curr_index, "layover_before"] = 0

    # Initialize new columns with original values of Arrival and Departure times
    df["Layover_adjusted_arrival_time"] = df["Adjusted Arrival Time"].apply(time_to_minutes)
    df["Layover_adjusted_departure_time"] = df["departure_time"].apply(time_to_minutes)

    # Apply layover adjustment for each vehicle
    for vehicle_id, group in df.groupby("Vehicle ID"):
        indices = group.index.tolist()
        layover_offset = 0  # Initialize layover time offset
        layover_started = False  # Flag to track if layover has started

        for i in range(len(indices)):
            curr_index = indices[i]

            if df.at[curr_index, "layover_before"] == 1:
                layover_offset = 360  # Add 360 minutes offset
                layover_started = True  # Mark that layover has started

            if layover_started:
                df.at[curr_index, "Layover_adjusted_arrival_time"] += layover_offset
                df.at[curr_index, "Layover_adjusted_arrival_time"] -= df.at[curr_index, "New Wait Time"]
                df.at[curr_index, "Layover_adjusted_departure_time"] += layover_offset
                df.at[curr_index, "Layover_adjusted_departure_time"] -= df.at[curr_index, "New Wait Time"]
            else:
                df.at[curr_index, "Layover_adjusted_arrival_time"] += layover_offset
                df.at[curr_index, "Layover_adjusted_departure_time"] += layover_offset

    # Adjust arrival time for rows after layover
    for vehicle_id, group in df.groupby("Vehicle ID"):
        indices = group.index.tolist()
        layover_started = False  # Flag to track when layover begins

        for i in range(len(indices)):
            curr_index = indices[i]

            # If layover_before = 1, layover starts from this row
            if df.at[curr_index, "layover_before"] == 1:
                layover_started = True  # Mark that layover has started
                continue  # Skip updating this row; changes apply to subsequent rows

            # If layover has already started, update arrival times for subsequent rows
            if layover_started and i > 0:
                prev_index = indices[i - 1]

                df.at[curr_index, "Layover_adjusted_arrival_time"] = (
                    df.at[prev_index, "Layover_adjusted_departure_time"] +
                    df.at[prev_index, "Travel Time to Next"]
                )

                df.at[curr_index, "Layover_adjusted_departure_time"] = (
                    df.at[curr_index, "Layover_adjusted_arrival_time"] +
                    df.at[curr_index, "Unloading Time"]
                )

    # Convert times back to HH:MM format
    df["Layover_adjusted_arrival_time"] = df["Layover_adjusted_arrival_time"].apply(minutes_to_time)
    df["Layover_adjusted_departure_time"] = df["Layover_adjusted_departure_time"].apply(minutes_to_time)


    return df[["Vehicle ID","Store ID","Location","Cube","Remaining Cube","Arrival Time","departure_time","Suggested Store Open Time","Adjusted Arrival Time","Minutes_of_service","layover_route","layover_after","layover_before","Layover_adjusted_arrival_time","Layover_adjusted_departure_time","original_wait_time","New Wait Time","Window Start","Window End","Distance to Next","Travel Time to Next","Wait Time","Unloading Time","Validation","Max Trailer Length (Store)","Vehicle Trailer Length","Store Preferred Vehicle Type","Vehicle Type","Dispatch Date"]]

def get_coordinates(final_df,coordinates_df):
    coordinates_df['Store'] = coordinates_df["id"].str.extract(r'(\d+)').fillna(0).astype(int).astype(str)
    coordinates_df = pd.concat([ pd.DataFrame({
        'id': ['DC'],
        'coordinate': ['(40.026330, -79.083500)'],
        'Store': ['DC']  # Ensure this matches the exact column name
    }),coordinates_df], ignore_index=True)
    coordinates_df['Store'] = coordinates_df['Store'].str.strip()
    final_df['Store ID'] = final_df['Store ID'].astype(str).str.strip()
    coordinates_df['Store'] = coordinates_df['Store'].astype(str).str.strip()
    full_final_df=pd.merge(final_df,coordinates_df[['Store','coordinate']],left_on='Store ID',right_on='Store',how='inner')
    return full_final_df


def distinct_lift_gate_types(text):
    if not isinstance(text, str):
        text = str(text)
    lift_gate_pattern = r"(maxon lift gate|reefer w/ liftgate|dry lift gate|reefer lift gate)"
    return ", ".join(re.findall(lift_gate_pattern, text))

def calculate_time(distance_matrix):
        speed_to_or_from_15501 = 60  # Speed for distances involving 15501
        speed_store_to_store = 40    # Speed for distances between stores
        hrs_to_minutes = 60          # Conversion factor for hours to minutes
        time_matrix = distance_matrix.copy()  # Create a copy of the matrix
        for i in distance_matrix.index:
            for j in distance_matrix.columns:
                if i == '15501' or j == '15501':  # Check if either row or column involves 15501
                    time_matrix.at[i, j] = round(distance_matrix.at[i, j] * hrs_to_minutes / speed_to_or_from_15501)
                else:  # For store-to-store distances
                    time_matrix.at[i, j] = round(distance_matrix.at[i, j] * hrs_to_minutes / speed_store_to_store)
        return time_matrix

# df_new = pd.DataFrame(data_new)

# Function to classify the store based on start and end time
def classify_store_time(start_time, end_time):
    start_hour = int(start_time.split(":")[0])
    end_hour = int(end_time.split(":")[0])
    
    if start_hour < 12 and end_hour <= 12:  # Both times are in AM
        return "AM Store"
    elif start_hour >= 12 and end_hour >= 12:  # Both times are in PM
        return "PM Store"
    else:  # Times span across AM and PM
        return "AM PM Store"



def generate_input_file(df):
    file_path = 'C:/Users/Dheerajnuka/Downloads/streamlit_route_app/utils/Service Locations All Stores Logistics Planning Export 3.xlsx'
    main_df = pd.read_excel(file_path)
    main_df['Equipment Type Restrictions']= main_df['Equipment Type Restrictions'].str.lower()
    
    # Updating the column to only contain valid "Lift Gate" types
    main_df['Lift_Gate_Types'] = main_df['Equipment Type Restrictions'].apply(distinct_lift_gate_types)

    # Filtering state with PA and city with Philadelphia
    main_df=main_df[["ID", "Description",'Window Open', 'Window Close', 'Coordinate','Country', 'State', 'City','Max Trailer Length','Equipment Type Restrictions']]

    # Convert column names to lowercase and replace spaces with underscores
    main_df.columns = [col.lower().replace(" ", "_") for col in main_df.columns]
    main_df.reset_index(drop=True)
    
    main_df['Store'] = main_df["id"].str.extract(r'(\d+)').fillna(0).astype(int)
    main_df = pd.concat([ pd.DataFrame({
        'id': ['15501'],
        'description': ['Somerset'],
        'window_open': ['00:00'],
        'window_close': ['23:59'],
        'coordinate': ['(40.026330, -79.083500)'],
        'country': ['United States'],
        'state': ['PA'],
        'city': ['Philadelphia'],
        'Store': [15501]  # Ensure this matches the exact column name
    }),main_df], ignore_index=True)
    
    df_input =df[['Store','Cube']]
    df_input=pd.concat([ pd.DataFrame({
        'Store': [15501],
        'Cube': [0]
    }),df_input], ignore_index=True)

    df_input.columns=['Store','Demand']

    main_df[main_df['Store'].isin(list(df_input['Store']))]
    main_df=main_df[main_df['Store'].isin(list(df_input['Store']))]

    
    # Extract coordinates from the 'coordinatge' column as tuples
    coordinates = main_df['coordinate'].str.extract(r'\(([^,]+),([^)]+)\)').astype(float).apply(tuple, axis=1).tolist()

    # Initialize an empty distance matrix DataFrame
    DistanceMatrix = pd.DataFrame(index=range(len(coordinates)), columns=range(len(coordinates)))

    # Calculate the geodesic distance between each pair of coordinates
    for i, j in itertools.product(range(len(coordinates)), repeat=2):
        if i != j:
            DistanceMatrix.iat[i, j] = geodesic(coordinates[i], coordinates[j]).miles
        else:
            DistanceMatrix.iat[i, j] = 0  # Distance from a point to itself


    # First, ensure we have as many stores as coordinates
    store_numbers = main_df['Store'][:len(coordinates)]

    # Reassign the index of the DistanceMatrix to 'Store' numbers
    DistanceMatrix.index = store_numbers
    DistanceMatrix.columns = store_numbers  # Update columns to have 'Store' labels as well


    # Set the name of the index and columns to None to remove "Store" duplication
    DistanceMatrix.index.name = None
    DistanceMatrix.columns.name = None
    TimeMatrix = calculate_time(DistanceMatrix)
    TimeMatrix.reset_index(col_level=0)
    time_matrix_df=TimeMatrix.reset_index(col_level=0)
    time_matrix_df=time_matrix_df[["index",15501]]
    time_matrix_df.columns=['store_id','distance']
    time_matrix_df["Haul Type"] = time_matrix_df["distance"].apply(lambda x: "Long Haul" if x >= 250 else "Short Haul")
    
    dmd_file_path = 'C:/Users/Dheerajnuka/Downloads/streamlit_route_app/utils/YTD Store Volume Average.xlsx'  # Placeholder path
    dmd_df = pd.read_excel(dmd_file_path, sheet_name='BY Store').drop(['AvgOfWeight', 'AvgOfPallet'], axis = 1)

    # Merge based on the 'ID' column (inner join by default)
    TimeWindows = pd.merge(main_df[['Store','window_open','window_close','max_trailer_length']], dmd_df[['Store']], on='Store', how='inner')  # Options: 'left', 'right', 'outer', 'inner'
    TimeWindows.columns = ['location', 'start_time','end_time','store_max_trailer_lengths']

    # add a record to TimeWindows
    TimeWindows = pd.concat([ pd.DataFrame({
        'location': [15501],
        'start_time': ['00:00'],
        'end_time': ['23:59'],
        'store_max_trailer_lengths':[100]
    }),TimeWindows], ignore_index=True)

    TimeWindows['lift_gate_types']="['reefer liftgate']"

    StoreDemands = df_input[['Store', 'Demand']]
    StoreDemands.columns = ['store_id','demand']

    def calculate_unloading_time(demand):
        if demand == 0:
            return 0
        elif demand > 0 and demand <= 500:
            return 30
        elif demand > 500 and demand <= 1000:
            return 60
        elif demand > 1000:
            return 75

    StoreDemands['unloading_time'] = StoreDemands['demand'].apply(calculate_unloading_time)
    data = {
        "vehicle_id": list(range(40)),
        "capacity": [1600] * 8 + [1800] * 8 + [2100] * 8 + [2600] * 16,
        "vehicle_trailer_lengths": [28] * 8 + [32] * 8 + [42] * 8 + [48] * 16,
        "vehicle_types": ["reefer liftgate"] * 40
    }
    VehicleCapacities = pd.DataFrame(data)
    VehicleCapacities['new_capacity'] = (VehicleCapacities['capacity'] * 0.95).astype(int)
    TimeWindows['Classification'] = TimeWindows.apply(lambda row: classify_store_time(row['start_time'], row['end_time']), axis=1)
        
    # Importing required library to save DataFrames to Excel
    from pandas import ExcelWriter

    file_path = 'file_input.xlsx'
    with ExcelWriter(file_path) as writer:
        DistanceMatrix.to_excel(writer, sheet_name='DistanceMatrix')
        TimeMatrix.to_excel(writer, sheet_name='TimeMatrix')
        TimeWindows.to_excel(writer, sheet_name='TimeWindows')
        StoreDemands.to_excel(writer, sheet_name='StoreDemands')
        VehicleCapacities.to_excel(writer, sheet_name='VehicleCapacities')
        time_matrix_df.to_excel(writer,sheet_name='houl_types')


    return DistanceMatrix,TimeMatrix,TimeWindows,StoreDemands,VehicleCapacities,time_matrix_df,main_df[['id','coordinate']]

def process_routes(df):
    DistanceMatrix,TimeMatrix,TimeWindows,StoreDemands,VehicleCapacities,time_matrix_df,coordinates_df=generate_input_file(df)
    # data=create_data_model_from_excel(DistanceMatrix,TimeMatrix,TimeWindows,StoreDemands,VehicleCapacities,time_matrix_df,num_vehicles=40)
    data=create_data_model_from_excel(num_vehicles=40)
    # print(data)
    data, solution, routing, manager, time_dimension=solve_routing_problem_with_unloading(data)

    if solution:
        for vehicle_id in range(data["num_vehicles"]):
            print(f"Route for vehicle {vehicle_id}:")
            index = routing.Start(vehicle_id)
            route = []
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
            print(" -> ".join(map(str, route)))
    else:
        print("No solution found!")

    # Print total cost
    # print(f"Total cost: {solution.ObjectiveValue()}")
    output_df=save_solution_to_excel(data, manager, routing, solution, time_dimension, output_file='vr_real_data_v1_Nov_1_updated.xlsx',
                           break_after_minutes=10000, break_duration_minutes=5, depot_id=15501, dispatch_date=None)
    # output_df.to_csv('output_df.csv',index=False)
    updated_df = add_suggested_store_open_time(output_df)
    updated_df["Store ID"] = updated_df["Store ID"].replace(15501, "DC")
    final_df=adjust_arrival_times(updated_df)
    full_final_df= layover_logics(final_df)
    vehicle_summary=vehicle_summary_calculation(final_df,VehicleCapacities)
    full_final_df1=get_coordinates(full_final_df,coordinates_df)
    # vehicle_summary.to_csv('vehicle_summary.csv',index=False)
    # full_final_df.to_csv('full_final_df.csv',index=False)
    return full_final_df1,vehicle_summary