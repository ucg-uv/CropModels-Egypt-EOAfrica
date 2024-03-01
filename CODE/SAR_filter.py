
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
from scipy.signal import savgol_filter
import numpy as np
import os
import pandas as pd
from datetime import datetime
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import geopandas as gpd

main_path = os.path.getcwd()

def get_date_from_string(date_string):
    # Parse the input string to a datetime object
    date_object = datetime.strptime(date_string, '%Y%m%d')

    return date_object


def get_index_date(dataframe, date):

    # Convert date to datetime
    date = datetime.strptime(date, '%Y%m%d')

    # Convert the date strings to datetime objects
    dataframe_dates = pd.to_datetime(dataframe['date'], format='%Y%m%d')

    # Calculate the absolute difference between each date in the column and the given date
    diff = abs(dataframe_dates - date)

    # Find the index of the minimum difference
    idx = diff.idxmin()

    return idx


def get_date_list(all_dates):
    dates_list = []
    for current_list in all_dates:
        dates = []
        for current_date in current_list:
            date = get_date_from_string(str(current_date))
            dates.append(date)
        dates_list.append(dates)

    return dates_list


def find_nearest_value_index(value, lst):
    return min(range(len(lst)), key=lambda i: abs(lst[i] - value))

csv_path = os.path.join(main_path,'OUTPUTS/GDD_SAR')
output_path = os.path.join(main_path,'OUTPUTS/filtered_SAR')

all_parcels_list = os.listdir(csv_path)

for parcel in range(10, 150):

    full_csv = pd.DataFrame()

    if str(parcel) + '.csv' in all_parcels_list:

        # Check year croptype and yield
        for year in range(2016, 2023):

            if year == 2018:

                SOS_wheat = '1005'
                EOS_wheat = '0501'
            else:

                SOS_wheat = '1015'
                EOS_wheat = '0501'

            # Read current parcel data
            current_csv = pd.read_csv(os.path.join(csv_path, str(parcel) + '.csv'))

            if len(current_csv) > 0:

                # Set SOS and EOS
                SOS = str(year - 1) + SOS_wheat
                EOS = str(year) + EOS_wheat

                # Get the index of SOS and EOS
                SOS_idx = get_index_date(current_csv, SOS)
                EOS_idx = get_index_date(current_csv, EOS)

                if not SOS_idx == EOS_idx:

                    # Save dates and ndvi
                    # current_csv['YEAR'] = current_csv['DATE'].astype(str).str[:4].astype(int)
                    filter_year = current_csv.iloc[SOS_idx:EOS_idx].reset_index()

                    if len(filter_year.date) > 0:

                        # Save just some columns

                        csv_dates = list(filter_year.date)
                        csv_season = [str(year)] * len(filter_year)
                        csv_gdd = list(filter_year.GDD)
                        csv_VV = list(filter_year.sigma0_VV)
                        csv_VH = list(filter_year.sigma0_VH)
                        csv_VVVH = list(filter_year['VH/VV'].values)

                        # Filter cloudy days

                        csv_dates_filtered = []
                        csv_season_filtered = []
                        csv_gdd_filtered = []
                        csv_VV_filtered = []
                        csv_VH_filtered = []
                        csv_VVVH_filtered = []

                        for element in range(0, len(csv_dates)):
                            csv_dates_filtered.append(csv_dates[element])

                            csv_season_filtered.append(csv_season[element])

                            csv_gdd_filtered.append(csv_gdd[element])
                            csv_VV_filtered.append(csv_VV[element])
                            csv_VH_filtered.append(csv_VH[element])
                            csv_VVVH_filtered.append(csv_VVVH[element])

                        # SG filter
                        winsize = 5
                        polyord = 2

                        if len(csv_dates_filtered) > winsize:
                            csv_golay_VV = savgol_filter(csv_VV_filtered, winsize, polyord)
                            csv_golay_VH = savgol_filter(csv_VH_filtered, winsize, polyord)
                            csv_golay_VVVH = savgol_filter(csv_VVVH_filtered, winsize, polyord)

                            # Interpolate GDD to get interval [0,50,100...3500]

                            # Create a new list of X values from 0 to 3700 with a step of 50
                            gdd_interpolated = np.arange(0, 3750, 50)

                            # Use linear interpolation to get new Y values

                            interpolator_VV = interp1d(csv_gdd_filtered, csv_golay_VV,
                                                       kind='linear',
                                                       fill_value="extrapolate")
                            interpolator_VH = interp1d(csv_gdd_filtered, csv_golay_VH,
                                                       kind='linear',
                                                       fill_value="extrapolate")
                            interpolator_VVVH = interp1d(csv_gdd_filtered, csv_golay_VVVH,
                                                         kind='linear',
                                                         fill_value="extrapolate")

                            interpolated_VV = list(interpolator_VV(gdd_interpolated))
                            interpolated_VH = list(interpolator_VH(gdd_interpolated))
                            interpolated_VVVH = list(interpolator_VVVH(gdd_interpolated))
                            gdd_interpolated = list(gdd_interpolated)

                            # ndvi_interpolated_precip = list(
                            #    interpolator_precip(precip_interpolated))
                            # precip_interpolated = list(precip_interpolated)

                            lenght = max(len(csv_dates_filtered), len(interpolated_VV),
                                         len(interpolated_VH),len(interpolated_VVVH) , len(gdd_interpolated))

                            csv_dates_filtered += [None] * (lenght - len(csv_dates_filtered))
                            csv_season_filtered += [csv_season_filtered[0]] * (
                                    lenght - len(csv_season_filtered))
                            csv_gdd_filtered += [None] * (lenght - len(csv_gdd_filtered))
                            gdd_interpolated += [None] * (lenght - len(gdd_interpolated))

                            interpolated_VV += [None] * (lenght - len(interpolated_VV))
                            interpolated_VH += [None] * (lenght - len(interpolated_VH))
                            interpolated_VVVH += [None] * (lenght - len(interpolated_VVVH))

                            csv_small = pd.DataFrame({'DATE': list(csv_dates_filtered),
                                                      'SEASON': list(csv_season_filtered),
                                                      'GDD': list(gdd_interpolated),
                                                      'GDD_raw': list(csv_gdd_filtered),
                                                      'VV': list(interpolated_VV),
                                                      'VH': list(interpolated_VH),
                                                      'VH/VV': list(interpolated_VVVH)})

                            full_csv = pd.concat([full_csv, csv_small], ignore_index=True)


        # Save full parcel
        if len(full_csv) > 5:
            full_csv.to_csv(os.path.join(output_path, str(parcel) + '.csv'), index=False)
            print(str(parcel))