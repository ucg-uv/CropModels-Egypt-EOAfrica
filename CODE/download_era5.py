import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

main_path = os.getcwd()
def date_range(start_date, end_date):
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += timedelta(days=1)

def datetime_to_string(dt, format_str="%Y%m%d"):
    return dt.strftime(format_str)


# Define the start and end date
start_date = datetime(2015, 9, 1)
end_date = datetime(2024, 1, 1)


# Authenticate to the Earth Engine servers
ee.Authenticate()

# Initialize the Earth Engine API
ee.Initialize()

# Define your study area geometry (a square of a determinated pixel)
# Region 1
#geometry = ee.Geometry.Rectangle([30.818045, 30.761451, 30.858884, 30.810041])
#point = ee.Geometry.Point(30.789622, 30.837803)

# Region 2
geometry = ee.Geometry.Rectangle([30.80850, 31.05403, 30.89055, 31.16105])
point = ee.Geometry.Point(31.10622, 30.84782)

full_df = []

i = 0
for date in date_range(start_date, end_date):

    # Define the ERA5 collection
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(date).filterBounds(geometry).first()

    val = era5.getInfo()

    if val is not None:

        pixel_value = era5.sample(point).first().getInfo()

        date_string = datetime_to_string(date)

        values_all = pixel_value['properties']
        values = {'temperature_2m': values_all['temperature_2m'], 'total_precipitation_sum': values_all['total_precipitation_sum']}

    else:

        date_string = datetime_to_string(date)
        values = {'temperature_2m': 0, 'total_precipitation_sum': 0}

    # Add an index:
    index = [i]
    i = i + 1

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(values, index=index)
    df.insert(0, 'date', date_string)
    full_df.append(df)


# Save the DataFrame to a CSV file
result_df = pd.concat(full_df, axis=0)

result_df.to_csv(os.path.join(main_path,'ERA5_2.csv'), index=False)

print("ERA5 data downloaded and saved to ERA5_2.csv.")
