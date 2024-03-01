#


import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
from scipy.signal import savgol_filter

main_path = os.path.getcwd()

def get_date_from_string(date_string):
    # Parse the input string to a datetime object
    date_object = datetime.strptime(date_string, '%Y%m%d')

    return date_object


def get_index_date(dataframe, date):

    # Convert date to datetime
    date = datetime.strptime(date, '%Y%m%d')

    # Convert the date strings to datetime objects
    dataframe_dates = pd.to_datetime(dataframe['DATE_inter'], format='%Y%m%d')

    # Calculate the absolute difference between each date in the column and the given date
    diff = abs(dataframe_dates - date)

    # Find the index of the minimum difference
    idx = diff.idxmin()

    return idx

def generate_dates(start_date, end_date, length):
    # Convert input strings to datetime objects
    start_date = datetime.strptime(start_date, "%Y%m%d")
    end_date = datetime.strptime(end_date, "%Y%m%d")

    # Calculate the step size
    step_size = (end_date - start_date) / (length - 1)

    # Generate the list of dates
    date_list = [start_date + i * step_size for i in range(length)]

    # Format the dates as strings
    formatted_dates = [date.strftime("%Y-%m-%d") for date in date_list]

    return formatted_dates

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


csv_path = os.path.join(main_path,'OUTPUTS/filtered_100')
output_path = os.path.join(main_path,'OUTPUTS/plots_100')

all_parcels_list = os.listdir(csv_path)


for croptype in ['wheat', 'rice']:
    plt.figure(figsize=(9, 6))
    Y_total_all = 0
    Y_total = 0
    incre = 0
    for year in [2016,2018,2019,2020,2021,2022,2023]:
        Y_total_all = Y_total_all + Y_total
        incre = incre + 0.05
        Y_total = 0

        if year == 2016:
            color = 'red'
        elif year == 2017:
            color = 'blue'
        elif year == 2018:
            color = 'green'
        elif year == 2019:
            color = 'black'
        elif year == 2020:
            color = 'purple'
        elif year == 2021:
            color = 'orange'
        elif year == 2022:
            color = 'cyan'
        elif year == 2023:
            color = 'brown'

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

        if croptype == 'wheat':
            # Set SOS and EOS
            SOS = str(year - 1) + SOS_wheat
            EOS = str(year) + EOS_wheat

        if croptype == 'rice':
            # Set SOS and EOS
            SOS = str(year) + SOS_rice
            EOS = str(year) + EOS_rice

        all_dates = []
        all_gdd = []
        all_ndvi = []
        all_yields = []
        all_smoothed = []
        all_golay = []
        all_ndvi_raw = []
        all_gdd_raw = []
        all_ndvi_inter = []
        all_dates_inter = []

        for parcel in range(1, 150):

            current_dates = []
            current_gdd = []
            current_ndvi = []
            current_yield = []
            current_smoothed = []
            current_golay = []

            if str(parcel) + '.csv' in all_parcels_list:

                current_csv = pd.read_csv(os.path.join(csv_path, str(parcel) + '.csv'))
                current_csv['SEASON'] = current_csv['SEASON'].fillna(np.nan).astype('Int64')
                current_csv['DATE'] = current_csv['DATE'].fillna(np.nan).astype('Int64')
                filtered_csv = current_csv[current_csv['SEASON'] == year].reset_index()
                filtered_csv = filtered_csv[filtered_csv['CROPTYPE'] == croptype].reset_index()

                if len(filtered_csv) > 0:
                    all_dates.append(filtered_csv['DATE'].dropna().tolist())
                    all_gdd.append(list(filtered_csv['GDD'].values))
                    all_ndvi.append(list(filtered_csv['NDVI'].values))
                    all_ndvi_inter.append(list(filtered_csv['NDVI_dates_inter'].values))
                    all_dates_inter.append(list(filtered_csv['DATE_inter'].values))
                    all_ndvi_raw.append(filtered_csv['NDVI_raw'].dropna().tolist())
                    all_yields.append(filtered_csv['YIELD'].dropna().tolist())
                    all_gdd_raw.append(filtered_csv['GDD_raw'].dropna().tolist())


        if len(all_dates) > 0:

            # Group by yields
            #plt.figure(figsize=(9,6))
            # Get unique yields
            first_yields = [sublist[0] for sublist in all_yields]
            unique_yields = np.unique(first_yields)


            for yield_group in range(0, len(unique_yields)):

                group_ndvi = []
                group_gdd = []
                group_data = []
                group_ndvi_raw = []
                group_gdd_raw = []
                group_ndvi_inter = []
                group_dates_inter = []

                for parcel in range(0, len(all_dates)):

                    if all_yields[parcel][0] == unique_yields[yield_group]:
                        group_ndvi.append(all_ndvi[parcel])
                        group_gdd.append(all_gdd[parcel])
                        group_data.append(all_dates[parcel])
                        group_ndvi_raw.append(all_ndvi_raw[parcel])
                        group_gdd_raw.append(all_gdd_raw[parcel])
                        group_ndvi_inter.append(all_ndvi_inter[parcel])
                        group_dates_inter.append(all_dates_inter[parcel])

                # Perform mean of yield group data
                #X = group_gdd
                #Y = group_ndvi
                X = group_dates_inter
                Y = group_ndvi_inter


                # Definir el vector entre 0 y 3000 con un paso de 100
                vector_X = generate_dates(SOS, EOS, 75)

                doy_list = []

                for date_str in vector_X:
                    date_object = datetime.strptime(date_str, "%Y-%m-%d")
                    day_of_year = date_object.timetuple().tm_yday
                    doy_list.append(day_of_year)


                means = []
                stdevs = []

                for x_dates in range(0, len(vector_X)):
                    values = []
                    for y_parcels in range(0, len(Y)):
                        values.append(Y[y_parcels][x_dates])
                    means.append(np.nanmean(values))
                    stdevs.append(np.nanstd(values))
                indexes = [0,4,9,14,19,24,29,34,39,44,49,54,50,64,69,74]

                x_final = [doy_list[i] for i in indexes]
                y_final = [means[i] for i in indexes]
                z_final = [stdevs[i] for i in indexes]


                # Graficar la curva promediada con barras de error
                #labell = '(' + str(len(Y)) + ') ' + str(np.round(unique_yields[yield_group] / 420, 2)) + ' t/ha'
                plt.errorbar(np.arange(len(x_final)), y_final, yerr=z_final, fmt='o-', capsize=5, color=color)
                #plt.plot(np.arange(len(x_final)), y_final)

                Y_total = Y_total + len(Y)

            plt.text(17, 0.6-incre, str(year) + ' (' + str(Y_total) + ')', color=color, fontsize=15, ha='left', va='center')

    plt.ylim(-0.1, 1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 0.55), fontsize='12')
    plt.xlabel('DOY')
    plt.ylabel('mean(NDVI)')
    plt.title('All seasons - Crop: ' + str(croptype) + ' - Nº parcels: ' + str(Y_total_all))
    plt.xticks(np.arange(len(x_final)), x_final, rotation=45)
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(output_path, 'all_doy_' + str(croptype) + '.png'),
                bbox_inches='tight')
    plt.close()



'''
#


import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
from scipy.signal import savgol_filter


def get_date_from_string(date_string):
    # Parse the input string to a datetime object
    date_object = datetime.strptime(date_string, '%Y%m%d')

    return date_object


def get_index_date(dataframe, date):

    # Convert date to datetime
    date = datetime.strptime(date, '%Y%m%d')

    # Convert the date strings to datetime objects
    dataframe_dates = pd.to_datetime(dataframe['DATE_inter'], format='%Y%m%d')

    # Calculate the absolute difference between each date in the column and the given date
    diff = abs(dataframe_dates - date)

    # Find the index of the minimum difference
    idx = diff.idxmin()

    return idx

def generate_dates(start_date, end_date, length):
    # Convert input strings to datetime objects
    start_date = datetime.strptime(start_date, "%Y%m%d")
    end_date = datetime.strptime(end_date, "%Y%m%d")

    # Calculate the step size
    step_size = (end_date - start_date) / (length - 1)

    # Generate the list of dates
    date_list = [start_date + i * step_size for i in range(length)]

    # Format the dates as strings
    formatted_dates = [date.strftime("%Y-%m-%d") for date in date_list]

    return formatted_dates

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


csv_path = os.path.join(main_path,'OUTPUTS/filtered_100')
output_path = os.path.join(main_path,'OUTPUTS/plots_100')

all_parcels_list = os.listdir(csv_path)


for croptype in ['wheat', 'rice']:

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

        if croptype == 'wheat':
            # Set SOS and EOS
            SOS = str(year - 1) + SOS_wheat
            EOS = str(year) + EOS_wheat

        if croptype == 'rice':
            # Set SOS and EOS
            SOS = str(year) + SOS_rice
            EOS = str(year) + EOS_rice

        all_dates = []
        all_gdd = []
        all_ndvi = []
        all_yields = []
        all_smoothed = []
        all_golay = []
        all_ndvi_raw = []
        all_gdd_raw = []
        all_ndvi_inter = []
        all_dates_inter = []

        for parcel in range(1, 150):

            current_dates = []
            current_gdd = []
            current_ndvi = []
            current_yield = []
            current_smoothed = []
            current_golay = []

            if str(parcel) + '.csv' in all_parcels_list:

                current_csv = pd.read_csv(os.path.join(csv_path, str(parcel) + '.csv'))
                current_csv['SEASON'] = current_csv['SEASON'].fillna(np.nan).astype('Int64')
                current_csv['DATE'] = current_csv['DATE'].fillna(np.nan).astype('Int64')
                filtered_csv = current_csv[current_csv['SEASON'] == year].reset_index()
                filtered_csv = filtered_csv[filtered_csv['CROPTYPE'] == croptype].reset_index()

                if len(filtered_csv) > 0:
                    all_dates.append(filtered_csv['DATE'].dropna().tolist())
                    all_gdd.append(list(filtered_csv['GDD'].values))
                    all_ndvi.append(list(filtered_csv['NDVI'].values))
                    all_ndvi_inter.append(list(filtered_csv['NDVI_dates_inter'].values))
                    all_dates_inter.append(list(filtered_csv['DATE_inter'].values))
                    all_ndvi_raw.append(filtered_csv['NDVI_raw'].dropna().tolist())
                    all_yields.append(filtered_csv['YIELD'].dropna().tolist())
                    all_gdd_raw.append(filtered_csv['GDD_raw'].dropna().tolist())


        if len(all_dates) > 0:

            # Group by yields
            plt.figure(figsize=(9,6))
            # Get unique yields
            first_yields = [sublist[0] for sublist in all_yields]
            unique_yields = np.unique(first_yields)

            Y_total = 0
            for yield_group in range(0, len(unique_yields)):

                group_ndvi = []
                group_gdd = []
                group_data = []
                group_ndvi_raw = []
                group_gdd_raw = []
                group_ndvi_inter = []
                group_dates_inter = []

                for parcel in range(0, len(all_dates)):

                    if all_yields[parcel][0] == unique_yields[yield_group]:
                        group_ndvi.append(all_ndvi[parcel])
                        group_gdd.append(all_gdd[parcel])
                        group_data.append(all_dates[parcel])
                        group_ndvi_raw.append(all_ndvi_raw[parcel])
                        group_gdd_raw.append(all_gdd_raw[parcel])
                        group_ndvi_inter.append(all_ndvi_inter[parcel])
                        group_dates_inter.append(all_dates_inter[parcel])

                # Perform mean of yield group data
                #X = group_gdd
                #Y = group_ndvi
                X = group_dates_inter
                Y = group_ndvi_inter


                # Definir el vector entre 0 y 3000 con un paso de 100
                vector_X = generate_dates(SOS, EOS, 75)

                doy_list = []

                for date_str in vector_X:
                    date_object = datetime.strptime(date_str, "%Y-%m-%d")
                    day_of_year = date_object.timetuple().tm_yday
                    doy_list.append(day_of_year)


                means = []
                stdevs = []

                for x_dates in range(0, len(vector_X)):
                    values = []
                    for y_parcels in range(0, len(Y)):
                        values.append(Y[y_parcels][x_dates])
                    means.append(np.nanmean(values))
                    stdevs.append(np.nanstd(values))
                indexes = [0,4,9,14,19,24,29,34,39,44,49,54,50,64,69,74]

                x_final = [doy_list[i] for i in indexes]
                y_final = [means[i] for i in indexes]
                z_final = [stdevs[i] for i in indexes]

                # Graficar la curva promediada con barras de error
                labell = '(' + str(len(Y)) + ') ' + str(np.round(unique_yields[yield_group] / 420, 2)) + ' t/ha'
                plt.errorbar(np.arange(len(x_final)), y_final, yerr=z_final, fmt='o-', label=labell, capsize=5)
                #plt.plot(np.arange(len(x_final)), y_final)

                Y_total = Y_total + len(Y)

            plt.ylim(-0.1, 1)
            plt.legend(loc='upper right', bbox_to_anchor=(1.35, 0.55), fontsize='12')
            plt.xlabel('DOY')
            plt.ylabel('mean(NDVI)')
            plt.title('Season: ' + str(year) + ' - Crop: ' + str(croptype) + ' - Nº parcels: ' + str(Y_total))
            plt.xticks(np.arange(len(x_final)), x_final)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'all_doy_' + str(croptype) + '.png'),
                        bbox_inches='tight')
            plt.close()

'''