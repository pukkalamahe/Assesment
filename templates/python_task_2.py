import pandas as pd
import numpy as np
import itertools


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    df = pd.read_csv('/content/dataset-3.csv')
    distances = {}
    
    for index, row in df.iterrows():
        from_location, to_location, distance = row['id_start'], row['id_end'], row['distance']
        if from_location not in distances:
            distances[from_location] = {}
        if to_location not in distances:
            distances[to_location] = {}
        distances[from_location][to_location] = distance
        distances[to_location][from_location] = distance 

    locations = sorted(distances.keys())
    df = pd.DataFrame(0, index=locations, columns=locations)
    for i in locations:
        for j in locations:
            if i != j:
                if j in distances[i]:
                    df.at[i, j] = distances[i][j]
                else:
                    intermediates = [k for k in locations if k != i and k != j and k in distances[i] and k in distances[j]]
                    min_distance = float('inf')
                    for intermediate in intermediates:
                        distance = distances[i][intermediate] + distances[intermediate][j]
                        if distance < min_distance:
                            min_distance = distance
                    if min_distance != float('inf'):
                        df.at[i, j] = min_distance
    return df


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    upper_triangle = distance_matrix.where(np.triu(np.ones(distance_matrix.shape), k=1).astype(bool))
    
    data = []
    for index, row in upper_triangle.iterrows():
        for column, value in row.iteritems():
            if not pd.isnull(value):
                data.append((index, column, value))
    
    df = pd.DataFrame(data, columns=['id_start', 'id_end', 'distance'])

    return df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    reference_df = df[df['id_start'] == reference_value]
    average_distance = reference_df['distance'].mean()
    threshold = 0.1 * average_distance
    close_ids = df[(df['id_start'] != reference_value) &
                   (df['distance'] >= (average_distance - threshold)) &
                   (df['distance'] <= (average_distance + threshold))]
    df = sorted(close_ids['id_start'].unique())

    return df


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}
    for vehicle, rate in rate_coefficients.items():
        df[vehicle] = df['distance'] * rate
    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here

    return df
