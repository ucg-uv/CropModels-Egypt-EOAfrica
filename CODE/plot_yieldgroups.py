# 5


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
    dataframe_dates = pd.to_datetime(dataframe['DATE'], format='%Y%m%d')

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


csv_path = os.path.join(main_path,'OUTPUTS/filtered_100')
output_path = os.path.join(main_path,'OUTPUTS/plots_100')

all_parcels_list = os.listdir(csv_path)



for croptype in ['wheat', 'rice']:

    if croptype == 'wheat':
        x_coor = 4100
    if croptype == 'rice':
        x_coor = 2500

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

        all_dates = []
        all_gdd = []
        all_ndvi = []
        all_yields = []
        all_smoothed = []
        all_golay = []
        all_ndvi_raw = []
        all_gdd_raw = []

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
                    all_ndvi_raw.append(filtered_csv['NDVI_raw'].dropna().tolist())
                    all_yields.append(filtered_csv['YIELD'].dropna().tolist())
                    all_gdd_raw.append(filtered_csv['GDD_raw'].dropna().tolist())


        if len(all_dates) > 0:

            # Group by yields

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

                for parcel in range(0, len(all_dates)):

                    if all_yields[parcel][0] == unique_yields[yield_group]:
                        group_ndvi.append(all_ndvi[parcel])
                        group_gdd.append(all_gdd[parcel])
                        group_data.append(all_dates[parcel])
                        group_ndvi_raw.append(all_ndvi_raw[parcel])
                        group_gdd_raw.append(all_gdd_raw[parcel])

                # Perform mean of yield group data
                #X = group_gdd
                #Y = group_ndvi
                X = group_gdd_raw
                Y = group_ndvi_raw


                # Definir el vector entre 0 y 3000 con un paso de 100
                if croptype == 'wheat':
                    vector_X = np.arange(0, 3800, 200)
                if croptype == 'rice':
                    vector_X = np.arange(0, 2300, 150)

                # Interpolar y promediar
                average_Y = np.zeros_like(vector_X, dtype=float)
                all_interpolations = []

                for x, y in zip(X, Y):
                    interpolated_Y = np.interp(vector_X, x, y)
                    average_Y += interpolated_Y
                    all_interpolations.append(interpolated_Y)

                average_Y /= len(X)  # Promedio

                # Calcular la desviación estándar
                std_deviation_Y = np.std(all_interpolations, axis=0)

                # Graficar la curva promediada con barras de error
                #labell = '(' + str(len(Y)) + ') ' + str(np.round(unique_yields[yield_group] / 420, 2)) + ' t/ha'
                plt.errorbar(vector_X, average_Y, yerr=std_deviation_Y, fmt='o-', color=color, capsize=5)

                Y_total = Y_total + len(Y)

            plt.text(x_coor, 0.6 - incre, str(year) + ' (' + str(Y_total) + ')', color=color, fontsize=15, ha='left',
                     va='center')

    plt.ylim(-0.1, 1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 0.55), fontsize='12')
    plt.xlabel('GDD')
    plt.ylabel('mean(NDVI)')
    plt.title('All seasons - Crop: ' + str(croptype) + ' - Nº parcels: ' + str(Y_total_all))
    plt.xticks(vector_X, rotation=45)

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(output_path, 'all_gdd_' + str(croptype) + '.png'),
                bbox_inches='tight')
    plt.close()


'''
# 5


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
    dataframe_dates = pd.to_datetime(dataframe['DATE'], format='%Y%m%d')

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


csv_path = os.path.join(main_path,'OUTPUTS/filtered_100')
output_path = os.path.join(main_path,'OUTPUTS/plots_100')

all_parcels_list = os.listdir(csv_path)


for croptype in ['wheat', 'rice']:

    for year in range(2016, 2023):

        all_dates = []
        all_gdd = []
        all_ndvi = []
        all_yields = []
        all_smoothed = []
        all_golay = []
        all_ndvi_raw = []
        all_gdd_raw = []

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

                for parcel in range(0, len(all_dates)):

                    if all_yields[parcel][0] == unique_yields[yield_group]:
                        group_ndvi.append(all_ndvi[parcel])
                        group_gdd.append(all_gdd[parcel])
                        group_data.append(all_dates[parcel])
                        group_ndvi_raw.append(all_ndvi_raw[parcel])
                        group_gdd_raw.append(all_gdd_raw[parcel])

                # Perform mean of yield group data
                #X = group_gdd
                #Y = group_ndvi
                X = group_gdd_raw
                Y = group_ndvi_raw


                # Definir el vector entre 0 y 3000 con un paso de 100
                if croptype == 'wheat':
                    vector_X = np.arange(0, 3750, 250)
                if croptype == 'rice':
                    vector_X = np.arange(0, 2250, 250)

                # Interpolar y promediar
                average_Y = np.zeros_like(vector_X, dtype=float)
                all_interpolations = []

                for x, y in zip(X, Y):
                    interpolated_Y = np.interp(vector_X, x, y)
                    average_Y += interpolated_Y
                    all_interpolations.append(interpolated_Y)

                average_Y /= len(X)  # Promedio

                # Calcular la desviación estándar
                std_deviation_Y = np.std(all_interpolations, axis=0)

                # Graficar la curva promediada con barras de error
                labell = '(' + str(len(Y)) + ') ' + str(np.round(unique_yields[yield_group] / 420, 2)) + ' t/ha'
                plt.errorbar(vector_X, average_Y, yerr=std_deviation_Y, fmt='o-', label=labell, capsize=5)
                Y_total = Y_total + len(Y)

            plt.ylim(-0.1, 1)
            plt.legend(loc='upper right', bbox_to_anchor=(1.35, 0.55), fontsize='12')
            plt.xlabel('GDD')
            plt.ylabel('mean(NDVI)')
            plt.title('Season: ' + str(year) + ' - Crop: ' + str(croptype) + ' - Nº parcels: ' + str(Y_total))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'yield_groups', str(croptype) + '_' + str(year) + '.png'),
                        bbox_inches='tight')


'''