import pandas as pd


def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Write your logic here
    # Input the data from dataset from csv file
    df = pd.read_csv('/content/dataset-1.csv')
    
    # Set columns, index and values from id_1, id_2 and car values from dataframe
    df = df.pivot(index='id_1', columns='id_2', values='car')

    # Manipulate the diagonal values as 0
    df = df.fillna(0)
    for i in range(min(df.shape)):
        df.iloc[i, i] = 0
    return df


def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Write your logic here

    df['car_type'] = pd.cut(df['car'], bins=[float('-inf'), 15, 25, float('inf')],
                              labels=['low', 'medium', 'high'])
    count = df['car_type'].value_counts().to_dict()
    return dict(sorted(count.items()))
    
    


def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Write your logic here
    mean_value = df['bus'].mean()
    result = df[df['bus'] > 2 * mean_value].index
    
    return list(result)


def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Write your logic here
    avg_truck_by_route = df.groupby('route')['truck'].mean()
    filtered_routes = avg_truck_by_route[avg_truck_by_route > 7].index
    
    return list(filtered_routes)


def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Write your logic here
    matrix = df.apptlymap(lambda x: x * 0.75 if x > 20 else x * 1.25)
    matrix = matrix.round(1)
    return matrix


def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    data['start_time'] = pd.to_datetime(data['startDay'] + ' ' + data['startTime'])
    
   
      data['end_time'] = pd.to_datetime(data['endDay'] + ' ' + data['endTime'])
      data['interval_duration'] = data['end_time'] - data['start_time']
      covers_24_hours = data['interval_duration'] == pd.Timedelta(days=1)
      covers_7_days = (data['start_time'].dt.dayofweek.min() == 0) & (data['end_time'].dt.dayofweek.max() == 6)
      data = ~(covers_24_hours & covers_7_days)
      series = completeness.groupby([data['id'], data['id_2']]).any()

      return series


df = pd.read_csv("datasets/dataset-1.csv")
print(generate_car_matrix(df))
print(get_type_count(df))
print(get_bus_indexes(df))
print(filter_routes(df))
print(multiply_matrix(matrix))
print(time_check(df))

