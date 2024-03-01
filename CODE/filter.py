# 3

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
import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime, timedelta

main_path = os.path.getcwd()


def interpolate_y_values(original_dates, original_y_values, start_date, end_date, output_length):
    # Convert integer dates to datetime objects
    original_dates = [datetime.strptime(str(date), '%Y%m%d') for date in original_dates]
    start_date = datetime.strptime(str(start_date), '%Y%m%d')
    end_date = datetime.strptime(str(end_date), '%Y%m%d')

    # Create an interpolation function
    interp_function = interp1d(
        [date.timestamp() for date in original_dates],
        original_y_values,
        kind='linear',
        fill_value='extrapolate'
    )

    # Calculate the step size based on the desired output length
    step = (end_date - start_date) / (output_length - 1)

    # Generate a list of dates with the calculated step size
    interpolate_dates = [start_date + i * step for i in range(output_length)]

    # Perform interpolation for the interpolate dates
    interpolated_y_values = interp_function([date.timestamp() for date in interpolate_dates])

    return [int(date.strftime('%Y%m%d')) for date in interpolate_dates], interpolated_y_values




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

csv_path = os.path.join(main_path,'OUTPUTS/GDD_NDVI_csv_100')
output_path = os.path.join(main_path,'/OUTPUTS/filtered_100')
all_parcels_list = os.listdir(csv_path)


rice_start = 1750
rice_end = 2500
wheat_start = 1250
wheat_end = 2250


for parcel in range(1, 150):

    full_csv = pd.DataFrame()

    if str(parcel) + '.csv' in all_parcels_list:


        # Check year croptype and yield
        for year in range(2016, 2023):


            if year == 2018:
                # SOS and EOS rice
                SOS_rice = '0601'
                EOS_rice = '0920'

                # SOS and EOS wheat
                SOS_wheat = '1005'
                EOS_wheat = '0501'
            else:
                # SOS and EOS rice
                SOS_rice = '0601'
                EOS_rice = '0920'

                # SOS and EOS wheat
                SOS_wheat = '1015'
                EOS_wheat = '0501'


            column_name = 'croptype_' + str(year)

            for crop in ['wheat', 'rice']:


                #if not (str(year) == '2017' and crop == 'wheat'):

                # Read current parcel data
                current_csv = pd.read_csv(os.path.join(csv_path, str(parcel) + '.csv'))

                # Get crop type
                croptype = current_csv[column_name].iloc[0]

                # Get reference yield
                raw_yield = current_csv['yield_' + str(year)].iloc[0]
                if isinstance(raw_yield, str) and raw_yield.count('-') > 0:
                    both_yields = [int(num) for num in raw_yield.split('-')]
                    if crop == 'wheat':
                        reference_yield = both_yields[0]
                    if crop == 'rice':
                        reference_yield = both_yields[1]
                else:
                    reference_yield = current_csv['yield_' + str(year)].iloc[0]

                reference_yield = np.float32(reference_yield)

                if isinstance(croptype, str) and ~np.isnan(reference_yield):

                    # Check if it is the correct croptype
                    if croptype.count(crop) > 0:

                        if crop == 'wheat':
                            # Set SOS and EOS
                            SOS = str(year - 1) + SOS_wheat
                            EOS = str(year) + EOS_wheat

                        if crop == 'rice':
                            # Set SOS and EOS
                            SOS = str(year) + SOS_rice
                            EOS = str(year) + EOS_rice

                        # Get the index of SOS and EOS
                        SOS_idx = get_index_date(current_csv, SOS)
                        EOS_idx = get_index_date(current_csv, EOS)

                        if not SOS_idx == EOS_idx:

                            # Save dates and ndvi
                            # current_csv['YEAR'] = current_csv['DATE'].astype(str).str[:4].astype(int)
                            filter_year = current_csv.iloc[SOS_idx:EOS_idx].reset_index()
                            filter_year['NDVI'] = ((filter_year['b8_mean'] - filter_year['b4_mean']) /
                                                   (filter_year['b8_mean'] + filter_year['b4_mean']))
                            filter_year['DVI'] = (filter_year['b8_mean'] - filter_year['b4_mean'])

                            if len(filter_year.date) >= 5:

                                # Apply Golay filtering
                                # filtered_ndvi = savgol_filter(filter_year.NDVI, 6, 2)
                                filtered_ndvi = list(filter_year.NDVI)
                                gdd = list(filter_year.GDD)

                                if crop == 'wheat':
                                    gdd_peak = 1750
                                if crop == 'rice':
                                    gdd_peak = 1750

                                # Filter peak
                                mid_ndvis = filtered_ndvi[find_nearest_value_index(gdd_peak - 300, gdd) : find_nearest_value_index(gdd_peak + 300, gdd)]
                                if any(element > 0.6 for element in mid_ndvis):
                                #if True:
                                    # Filter mins, len and max logical
                                    if min(gdd) > 0 and max(gdd) < 10000 and len(gdd) > 5:

                                        # Filter some max value threshold
                                        if max(filtered_ndvi) > 0.4:

                                            # Filter the highest initial NDVIs
                                            #if len(filtered_ndvi[:find_nearest_value_index(500, gdd)]) > 0:
                                            #if True:
                                            if filtered_ndvi[find_nearest_value_index(100, gdd)] < 0.4:
                                                #if max(filtered_ndvi[:find_nearest_value_index(500, gdd)]) < 0.4:
                                                if filtered_ndvi[find_nearest_value_index(3500, gdd)] < 0.4:
                                                #if True:

                                                    # Save just some columns
                                                    csv_ndvi = list(filter_year.NDVI)
                                                    csv_dvi = list(filter_year.DVI)
                                                    csv_dates = list(filter_year.date)
                                                    csv_dates_int = list(filter_year.date.astype(int))
                                                    csv_yield = [str(reference_yield)] * len(filter_year)
                                                    csv_season = [str(year)] * len(filter_year)
                                                    csv_croptype = [str(crop)] * len(filter_year)
                                                    csv_gdd = list(filter_year.GDD)

                                                    csv_b1 = list(filter_year.b1_mean)
                                                    csv_b2 = list(filter_year.b2_mean)
                                                    csv_b3 = list(filter_year.b3_mean)
                                                    csv_b4 = list(filter_year.b4_mean)
                                                    csv_b5 = list(filter_year.b5_mean)
                                                    csv_b6 = list(filter_year.b6_mean)
                                                    csv_b7 = list(filter_year.b7_mean)
                                                    csv_b8 = list(filter_year.b8_mean)
                                                    csv_b8a = list(filter_year.b8a_mean)
                                                    csv_b9 = list(filter_year.b9_mean)
                                                    csv_b11 = list(filter_year.b11_mean)
                                                    csv_b12 = list(filter_year.b12_mean)

                                                    # Filter cloudy days
                                                    csv_ndvi_filtered = []
                                                    csv_dvi_filtered = []
                                                    csv_dates_filtered = []
                                                    csv_yield_filtered = []
                                                    csv_season_filtered = []
                                                    csv_croptype_filtered = []
                                                    csv_gdd_filtered = []

                                                    csv_b1_filtered = []
                                                    csv_b2_filtered = []
                                                    csv_b3_filtered = []
                                                    csv_b4_filtered = []
                                                    csv_b5_filtered = []
                                                    csv_b6_filtered = []
                                                    csv_b7_filtered = []
                                                    csv_b8_filtered = []
                                                    csv_b8a_filtered = []
                                                    csv_b9_filtered = []
                                                    csv_b11_filtered = []
                                                    csv_b12_filtered = []

                                                    if crop == 'wheat':
                                                        start = wheat_start
                                                        end = wheat_end
                                                    else:
                                                        start = rice_start
                                                        end = rice_end

                                                    for element in range(0, len(csv_dates)):

                                                        csv_ndvi_filtered.append(csv_ndvi[element])
                                                        csv_dvi_filtered.append(csv_dvi[element])
                                                        csv_dates_filtered.append(csv_dates[element])

                                                        csv_yield_filtered.append(csv_yield[element])
                                                        csv_season_filtered.append(csv_season[element])
                                                        csv_croptype_filtered.append(csv_croptype[element])
                                                        csv_gdd_filtered.append(csv_gdd[element])

                                                        csv_b1_filtered.append(csv_b1[element])
                                                        csv_b2_filtered.append(csv_b2[element])
                                                        csv_b3_filtered.append(csv_b3[element])
                                                        csv_b4_filtered.append(csv_b4[element])
                                                        csv_b5_filtered.append(csv_b5[element])
                                                        csv_b6_filtered.append(csv_b6[element])
                                                        csv_b7_filtered.append(csv_b7[element])
                                                        csv_b8_filtered.append(csv_b8[element])
                                                        csv_b8a_filtered.append(csv_b8a[element])
                                                        csv_b9_filtered.append(csv_b9[element])
                                                        csv_b11_filtered.append(csv_b11[element])
                                                        csv_b12_filtered.append(csv_b12[element])

                                                    # SG filter
                                                    winsize = 5
                                                    polyord = 2

                                                    if len(csv_ndvi_filtered) > winsize:

                                                        csv_ndvi = savgol_filter(csv_ndvi_filtered, winsize, polyord)
                                                        csv_dvi = savgol_filter(csv_dvi_filtered, winsize, polyord)

                                                        csv_golay_b1 = savgol_filter(csv_b1_filtered, winsize, polyord)
                                                        csv_golay_b2 = savgol_filter(csv_b2_filtered, winsize, polyord)
                                                        csv_golay_b3 = savgol_filter(csv_b3_filtered, winsize, polyord)
                                                        csv_golay_b4 = savgol_filter(csv_b4_filtered, winsize, polyord)
                                                        csv_golay_b5 = savgol_filter(csv_b5_filtered, winsize, polyord)
                                                        csv_golay_b6 = savgol_filter(csv_b6_filtered, winsize, polyord)
                                                        csv_golay_b7 = savgol_filter(csv_b7_filtered, winsize, polyord)
                                                        csv_golay_b8 = savgol_filter(csv_b8_filtered, winsize, polyord)
                                                        csv_golay_b8a = savgol_filter(csv_b8a_filtered, winsize,
                                                                                      polyord)
                                                        csv_golay_b9 = savgol_filter(csv_b9_filtered, winsize, polyord)
                                                        csv_golay_b11 = savgol_filter(csv_b11_filtered, winsize,
                                                                                      polyord)
                                                        csv_golay_b12 = savgol_filter(csv_b12_filtered, winsize,
                                                                                      polyord)

                                                        # Interpolate GDD to get interval [0,50,100...3500]

                                                        # Create a new list of X values from 0 to 3500 with a step of 50
                                                        gdd_interpolated = np.arange(0, 3750, 50)
                                                        # precip_interpolated = np.arange(0, 4.97, 0.07)

                                                        # Use linear interpolation to get new Y values

                                                        interpolator_ndvi = interp1d(csv_gdd_filtered, csv_ndvi,
                                                                                     kind='linear',
                                                                                     fill_value="extrapolate")
                                                        interpolator_dvi = interp1d(csv_gdd_filtered, csv_dvi,
                                                                                    kind='linear',
                                                                                    fill_value="extrapolate")
                                                        interpolator_b1 = interp1d(csv_gdd_filtered, csv_golay_b1,
                                                                                   kind='linear',
                                                                                   fill_value="extrapolate")
                                                        interpolator_b2 = interp1d(csv_gdd_filtered, csv_golay_b2,
                                                                                   kind='linear',
                                                                                   fill_value="extrapolate")
                                                        interpolator_b3 = interp1d(csv_gdd_filtered, csv_golay_b3,
                                                                                   kind='linear',
                                                                                   fill_value="extrapolate")
                                                        interpolator_b4 = interp1d(csv_gdd_filtered, csv_golay_b4,
                                                                                   kind='linear',
                                                                                   fill_value="extrapolate")
                                                        interpolator_b5 = interp1d(csv_gdd_filtered, csv_golay_b5,
                                                                                   kind='linear',
                                                                                   fill_value="extrapolate")
                                                        interpolator_b6 = interp1d(csv_gdd_filtered, csv_golay_b6,
                                                                                   kind='linear',
                                                                                   fill_value="extrapolate")
                                                        interpolator_b7 = interp1d(csv_gdd_filtered, csv_golay_b7,
                                                                                   kind='linear',
                                                                                   fill_value="extrapolate")
                                                        interpolator_b8 = interp1d(csv_gdd_filtered, csv_golay_b8,
                                                                                   kind='linear',
                                                                                   fill_value="extrapolate")
                                                        interpolator_b8a = interp1d(csv_gdd_filtered, csv_golay_b8a,
                                                                                    kind='linear',
                                                                                    fill_value="extrapolate")
                                                        interpolator_b9 = interp1d(csv_gdd_filtered, csv_golay_b9,
                                                                                   kind='linear',
                                                                                   fill_value="extrapolate")
                                                        interpolator_b11 = interp1d(csv_gdd_filtered, csv_golay_b11,
                                                                                    kind='linear',
                                                                                    fill_value="extrapolate")
                                                        interpolator_b12 = interp1d(csv_gdd_filtered, csv_golay_b12,
                                                                                    kind='linear',
                                                                                    fill_value="extrapolate")

                                                        # interpolator_precip = interp1d(precip_list, ndvi_smooth,
                                                        #                               kind='linear',
                                                        #                               fill_value="extrapolate")

                                                        ndvi_interpolated = list(interpolator_ndvi(gdd_interpolated))
                                                        dvi_interpolated = list(interpolator_dvi(gdd_interpolated))

                                                        interpolated_b1 = list(interpolator_b1(gdd_interpolated))
                                                        interpolated_b2 = list(interpolator_b2(gdd_interpolated))
                                                        interpolated_b3 = list(interpolator_b3(gdd_interpolated))
                                                        interpolated_b4 = list(interpolator_b4(gdd_interpolated))
                                                        interpolated_b5 = list(interpolator_b5(gdd_interpolated))
                                                        interpolated_b6 = list(interpolator_b6(gdd_interpolated))
                                                        interpolated_b7 = list(interpolator_b7(gdd_interpolated))
                                                        interpolated_b8 = list(interpolator_b8(gdd_interpolated))
                                                        interpolated_b8a = list(interpolator_b8a(gdd_interpolated))
                                                        interpolated_b9 = list(interpolator_b9(gdd_interpolated))
                                                        interpolated_b11 = list(interpolator_b11(gdd_interpolated))
                                                        interpolated_b12 = list(interpolator_b12(gdd_interpolated))
                                                        gdd_interpolated = list(gdd_interpolated)

                                                        # ndvi_interpolated_precip = list(
                                                        #    interpolator_precip(precip_interpolated))
                                                        # precip_interpolated = list(precip_interpolated)

                                                        lenght = max(len(csv_dates_filtered), len(dvi_interpolated),
                                                                     len(ndvi_interpolated), len(interpolated_b1),
                                                                     len(interpolated_b2), len(interpolated_b3),
                                                                     len(interpolated_b4), len(interpolated_b5),
                                                                     len(interpolated_b6), len(interpolated_b7),
                                                                     len(interpolated_b8), len(interpolated_b8a),
                                                                     len(interpolated_b9), len(interpolated_b11),
                                                                     len(interpolated_b12))

                                                        interpolated_dates, interpolated_ndvi_dates = interpolate_y_values(csv_dates_filtered, csv_ndvi_filtered, int(SOS), int(EOS), lenght)

                                                        csv_dates_filtered += [None] * (lenght - len(csv_dates_filtered))
                                                        csv_season_filtered += [csv_season_filtered[0]] * (lenght - len(csv_season_filtered))
                                                        csv_croptype_filtered += [csv_croptype_filtered[0]] * (lenght - len(csv_croptype_filtered))
                                                        csv_ndvi_filtered += [None] * (lenght - len(csv_ndvi_filtered))
                                                        csv_gdd_filtered += [None] * (lenght - len(csv_gdd_filtered))
                                                        gdd_interpolated += [None] * (lenght - len(gdd_interpolated))
                                                        csv_yield_filtered += [None] * (lenght - len(csv_yield_filtered))

                                                        ndvi_interpolated += [None] * (lenght - len(ndvi_interpolated))
                                                        dvi_interpolated += [None] * (lenght - len(dvi_interpolated))

                                                        interpolated_b1 += [None] * (lenght - len(interpolated_b1))
                                                        interpolated_b2 += [None] * (lenght - len(interpolated_b2))
                                                        interpolated_b3 += [None] * (lenght - len(interpolated_b3))
                                                        interpolated_b4 += [None] * (lenght - len(interpolated_b4))
                                                        interpolated_b5 += [None] * (lenght - len(interpolated_b5))
                                                        interpolated_b6 += [None] * (lenght - len(interpolated_b6))
                                                        interpolated_b7 += [None] * (lenght - len(interpolated_b7))
                                                        interpolated_b8 += [None] * (lenght - len(interpolated_b8))
                                                        interpolated_b8a += [None] * (lenght - len(interpolated_b8a))
                                                        interpolated_b9 += [None] * (lenght - len(interpolated_b9))
                                                        interpolated_b11 += [None] * (lenght - len(interpolated_b11))
                                                        interpolated_b12 += [None] * (lenght - len(interpolated_b12))

                                                        csv_small = pd.DataFrame({'DATE': list(csv_dates_filtered),
                                                                                  'DATE_inter': list(interpolated_dates),
                                                                                  'SEASON': list(csv_season_filtered),
                                                                                  'CROPTYPE': list(csv_croptype_filtered),
                                                                                  'NDVI_raw': list(csv_ndvi_filtered),
                                                                                  'NDVI': list(ndvi_interpolated),
                                                                                  'NDVI_dates_inter': list(interpolated_ndvi_dates),
                                                                                  'DVI': list(dvi_interpolated),
                                                                                  'GDD': list(gdd_interpolated),
                                                                                  'GDD_raw': list(csv_gdd_filtered),
                                                                                  'YIELD': list(csv_yield_filtered),
                                                                                  'B1': list(interpolated_b1),
                                                                                  'B2': list(interpolated_b2),
                                                                                  'B3': list(interpolated_b3),
                                                                                  'B4': list(interpolated_b4),
                                                                                  'B5': list(interpolated_b5),
                                                                                  'B6': list(interpolated_b6),
                                                                                  'B7': list(interpolated_b7),
                                                                                  'B8': list(interpolated_b8),
                                                                                  'B8A': list(interpolated_b8a),
                                                                                  'B9': list(interpolated_b9),
                                                                                  'B11': list(interpolated_b11),
                                                                                  'B12': list(interpolated_b12)})

                                                        full_csv = pd.concat([full_csv, csv_small], ignore_index=True)


        # Save full parcel
        if len(full_csv) > 5:
            full_csv.to_csv(os.path.join(output_path, str(parcel) + '.csv'), index=False)
