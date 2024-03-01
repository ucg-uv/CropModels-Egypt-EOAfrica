import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import geopandas as gpd

main_path = os.path.getcwd()

SAR_path = os.path.join(main_path,'Sentinel1/GRD_dataframes_EO_africa/')
output_path = os.path.join(main_path,'INPUTS/SAR')

for parcel in range(1, 150):

    full_parcel = []

    for year in range(2017, 2023):
        current_SAR = gpd.read_file(os.path.join(SAR_path, 'GRD-Yield_dataframe-wheat-' + str(year) + '.geojson'))
        current_SAR = pd.DataFrame(current_SAR)
        current_SAR['fid'] = current_SAR['fid'].astype(int)
        current_parcel = current_SAR[current_SAR['fid'] == parcel].reset_index(drop=True)
        full_parcel.append(current_parcel)

    export_parcel = pd.DataFrame(pd.concat(full_parcel))
    export_parcel['date'] = export_parcel['date'].dt.strftime('%Y%m%d')
    export_parcel = export_parcel.sort_values(by=['date'])
    export_parcel.to_csv(os.path.join(output_path, str(parcel) + '.csv'), index=False)
    print(str(parcel) + '.csv saved')


