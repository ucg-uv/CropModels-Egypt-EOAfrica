# 2

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import geopandas as gpd

main_path = os.path.getcwd()

def date_range(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, '%Y%m%d')
    end_date = datetime.strptime(end_date_str, '%Y%m%d')

    current_date = start_date
    while current_date <= end_date:
        yield current_date.strftime('%Y%m%d')
        current_date += timedelta(days=1)


parcels_1 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '55', '56', '57', '58', '59', '60', '61',
 '62', '63', '64', '65', '66', '67', '68', '69', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84',
 '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103',
 '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120',
 '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137',
 '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149']


# ERA5 data
ERA5_1 = pd.read_csv(os.path.join(main_path,'INPUTS/ERA5/ERA5_1.csv'))
ERA5_1['dates_str'] = ERA5_1['date'].astype(str)
ERA5_2 = pd.read_csv(os.path.join(main_path,'INPUTS/ERA5/ERA5_2.csv'))
ERA5_2['dates_str'] = ERA5_2['date'].astype(str)


# SOS and EOS rice
SOS_rice = '0601'
EOS_rice = '1001'

# SOS and EOS wheat
SOS_wheat = '1020'
EOS_wheat = '0501'

# Polys with SOS
polys_1 = gpd.read_file(os.path.join(main_path,'INPUTS/SOS_WorldCereal/polys_1.gpkg'))
polys_2 = gpd.read_file(os.path.join(main_path,'INPUTS/SOS_WorldCereal/polys_2.gpkg'))

# SAR path
SAR_path = os.path.join(main_path,'INPUTS/SAR')

# S2 path
S2_path = os.path.join(main_path,'OUTPUTS/Reflects_100')
all_parcels_list = os.listdir(S2_path)

# Output path
output_path = os.path.join(main_path,'OUTPUTS/GDD_NDVI_csv_100')

for parcel in range(1, 150):

    if str(parcel) + '.csv' in all_parcels_list:

        current_parcel = pd.read_csv(os.path.join(S2_path, str(parcel) + '.csv'))
        current_parcel['GDD'] = np.zeros(len(current_parcel), dtype=np.float64)
        current_parcel['GDD'] = current_parcel['GDD'].astype(np.float64)


        if parcels_1.count(str(parcel)) > 0:
            ERA5 = ERA5_1
        else:
            ERA5 = ERA5_2

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


            croptype = current_parcel['croptype_' + str(year)].iloc[0]

            # Loop between dates
            if croptype == 'wheat':

                gdd = 0

                SOS = str(year-1) + SOS_wheat
                EOS = str(year) + EOS_wheat

                index_start = ERA5[ERA5['dates_str'] == SOS].index[0]
                index_end = ERA5[ERA5['dates_str'] == EOS].index[0]

                # Loop between dates
                for date in date_range(SOS, EOS):

                    if date in ERA5['dates_str'].values:
                        index = ERA5[ERA5['dates_str'] == date].index[0]
                    else:
                        index = -1

                    if index != -1:
                        temp = ERA5['temperature_2m'].iloc[index] - 273.15
                        if temp < 0:
                            temp = 0
                        gdd = np.float64(gdd) + np.float64(temp)

                    if date in current_parcel['date'].values.astype(str):
                        index = current_parcel[current_parcel['date'].astype(str) == date].index[0]
                    else:
                        index = -1

                    if index != -1:
                        current_parcel.loc[index, 'GDD'] = gdd

            if croptype == 'rice':

                gdd = 0

                SOS = str(year) + SOS_rice
                EOS = str(year) + EOS_rice

                index_start = ERA5[ERA5['dates_str'] == SOS].index[0]
                index_end = ERA5[ERA5['dates_str'] == EOS].index[0]

                # Loop between dates
                for date in date_range(SOS, EOS):

                    if date in ERA5['dates_str'].values:
                        index = ERA5[ERA5['dates_str'] == date].index[0]
                    else:
                        index = -1

                    if index != -1:
                        temp = ERA5['temperature_2m'].iloc[index] - 283.15
                        if temp < 0:
                            temp = 0
                        gdd = np.float64(gdd) + np.float64(temp)

                    if date in current_parcel['date'].values.astype(str):
                        index = current_parcel[current_parcel['date'].astype(str) == date].index[0]
                    else:
                        index = -1

                    if index != -1:
                        current_parcel.loc[index, 'GDD'] = gdd

            if croptype == 'wheat-rice' or croptype == 'rice-wheat' or croptype == 'wheat - rice' or croptype == 'rice - wheat':

                crop = ['wheat', 'rice']

                for current_crop in crop:

                    if current_crop == 'wheat':

                        SOS = str(year-1) + SOS_wheat
                        EOS = str(year) + EOS_wheat

                        gdd = 0

                        index_start = ERA5[ERA5['dates_str'] == SOS].index[0]
                        index_end = ERA5[ERA5['dates_str'] == EOS].index[0]

                        # Loop between dates
                        for date in date_range(SOS, EOS):

                            if date in ERA5['dates_str'].values:
                                index = ERA5[ERA5['dates_str'] == date].index[0]
                            else:
                                index = -1

                            if index != -1:
                                temp = ERA5['temperature_2m'].iloc[index] - 273.15
                                if temp < 0:
                                    temp = 0
                                gdd = np.float64(gdd) + np.float64(temp)

                            if date in current_parcel['date'].values.astype(str):
                                index = current_parcel[current_parcel['date'].astype(str) == date].index[0]
                            else:
                                index = -1

                            if index != -1:
                                current_parcel.loc[index, 'GDD'] = gdd

                    if current_crop == 'rice':
                        SOS = str(year) + SOS_rice
                        EOS = str(year) + EOS_rice


                        gdd = 0

                        index_start = ERA5[ERA5['dates_str'] == SOS].index[0]
                        index_end = ERA5[ERA5['dates_str'] == EOS].index[0]

                        # Loop between dates
                        for date in date_range(SOS, EOS):

                            if date in ERA5['dates_str'].values:
                                index = ERA5[ERA5['dates_str'] == date].index[0]
                            else:
                                index = -1

                            if index != -1:
                                temp = ERA5['temperature_2m'].iloc[index] - 283.15
                                if temp < 0:
                                    temp = 0
                                gdd = gdd + temp

                            if date in current_parcel['date'].values.astype(str):
                                index = current_parcel[current_parcel['date'].astype(str) == date].index[0]
                            else:
                                index = -1

                            if index != -1:
                                current_parcel.loc[index, 'GDD'] = gdd


        print(str(parcel))
        current_parcel.to_csv(os.path.join(output_path, str(parcel) + '.csv'), index=False)