# 1
import os
import sys
import geopandas as gpd
import numpy as np
import pandas as pd
from glob import glob
import rasterio as rs
from rasterio import mask
from shapely.geometry import Point, LineString, Polygon, GeometryCollection
from shapely.ops import unary_union
import sys
from shapely.geometry import Polygon
import geopandas as gpd
from rasterio.mask import mask
import os
from shapely.geometry import mapping
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from rasterio.enums import Resampling
import rasterio
from rasterio.mask import mask


def cut_raster_poly(raster_path, geometry):
    with rasterio.open(raster_path) as src:
        # Crop the raster using the reprojected MultiPolygon
        crop, transform = mask(src, geometry, crop=True)
        # profile = src.profile
        return crop


columns_data = ['date', 'lat', 'long', 'parcel']
columns_crops = ['croptype_2016', 'croptype_2017', 'croptype_2018', 'croptype_2019',
                 'croptype_2020', 'croptype_2021', 'croptype_2022']
columns_yields = ['yield_2016', 'yield_2017', 'yield_2018', 'yield_2019',
                  'yield_2020', 'yield_2021', 'yield_2022']

columns_means = ['b1_mean', 'b2_mean', 'b3_mean', 'b4_mean', 'b5_mean', 'b6_mean', 'b7_mean', 'b8_mean',
                 'b8a_mean', 'b9_mean', 'b10_mean', 'b11_mean', 'b12_mean', 'bQA_mean']

columns_stdev = ['b1_std', 'b2_std', 'b3_std', 'b4_std', 'b5_std', 'b6_std', 'b7_std', 'b8_std',
                 'b8a_std', 'b9_std', 'b10_std', 'b11_std', 'b12_std', 'bQA_std']

columns_mins = ['b1_min', 'b2_min', 'b3_min', 'b4_min', 'b5_min', 'b6_min', 'b7_min', 'b8_min',
                'b8a_min', 'b9_min', 'b10_min', 'b11_min', 'b12_min', 'bQA_min']
columns_maxs = ['b1_max', 'b2_max', 'b3_max', 'b4_max', 'b5_max', 'b6_max', 'b7_max', 'b8_max',
                'b8a_max', 'b9_max', 'b10_max', 'b11_max', 'b12_max', 'bQA_max']
columns_medians = ['b1_median', 'b2_median', 'b3_median', 'b4_median', 'b5_median', 'b6_median', 'b7_median',
                   'b8_median', 'b8a_median', 'b9_median', 'b10_median', 'b11_median', 'b12_median', 'bQA_median']
columns_indexes = ['NDVI', 'NDVI_std', 'DVI', 'DVI_std']
columns_geoms = ['geometry_reprojected', 'geometry_original']



main_path = os.getcwd()
polys_1 = gpd.read_file(os.path.join(main_path,'polys_1.gpkg'))
polys_2 = gpd.read_file(os.path.join(main_path,'polys_2.gpkg'))

output_path = os.path.join(main_path,'OUTPUTS/Reflects_30')


for region in ['1', '2']:

    if region == '1':
        polys = polys_1
        current_tile = 'T36RTV'
    if region == '2':
        polys = polys_2
        current_tile = 'T36RUV'

    all_S2_path = os.path.join('/S2s/EOAfrica_' + region)

    for parcel in range(0, len(polys)):

        parcel_data = []
        parcel_means = []
        parcel_stds = []
        parcel_calcs = []
        parcel_mins = []
        parcel_maxs = []
        parcel_medians = []
        parcel_crops = []
        parcel_yields = []
        parcel_indexes = []
        parcel_geoms = []

        # Get current geometry
        current_polygon = polys.iloc[parcel]
        current_geometry = current_polygon['geometry']

        # List of all available images
        S2_dates = glob(os.path.join(all_S2_path, 'L1C_' + str(current_tile) + '*'))

        # Loop all dates
        for S2_date in S2_dates:

            all_S2_im_path = glob(os.path.join(S2_date, 'srLaSRCS2AV3.5.7-L1C_' + str(current_tile) + '*.tif'))

            # Choose the correct tiff
            for element in all_S2_im_path:

                if (element.count('BASK') == 0 and element.count('QMA_v2') == 0):

                    S2_im_path = element
                    #mask_path = element[:-4] + '_BASK.tif'
                    mask_path = glob(S2_date + '/L1C_' + str(current_tile) + '*_CLOUDLESS_30.tif')[0]

                    S2_im = rs.open(S2_im_path)

                    # Reproject the MultiPolygon to the S2 projection
                    # Put multipolygon into a geoseries and then iloc[0]
                    multipolygon = gpd.GeoDataFrame(geometry=[current_geometry], crs='EPSG:4326')
                    multipolygon_reprojected = multipolygon.to_crs('EPSG:32636')

                    # Crop im and mask
                    S2_cropped = cut_raster_poly(S2_im_path, multipolygon_reprojected.iloc[0])
                    mask_cropped = cut_raster_poly(mask_path, multipolygon_reprojected.iloc[0])

                    # Set background to nan
                    test_band = S2_cropped[7]
                    test_band_2 = S2_cropped[3]
                    test_band_3 = S2_cropped[12]
                    S2_cropped = np.where(S2_cropped == 0, np.nan, S2_cropped)

                    # Number of non cloud pixels
                    n_pixels = max(len(test_band[test_band != 0]), len(test_band_2[test_band_2 != 0]), len(test_band_3[test_band_3 != 0]))

                    # Only consider big parcels
                    if n_pixels > 2:

                        # Mask the S2 im with Bertran's BASK mask
                        filtered_im = S2_cropped * mask_cropped

                        # Set a -9999999 value where masks says
                        filtered_im = np.where(filtered_im == 0, -9999999, filtered_im)

                        filtered_im = filtered_im / 10000

                        # -9999999 values are clouds, get the cloud ratio
                        cloudy_pixels = len(filtered_im[filtered_im < 0])
                        clear_pixels = len(filtered_im[filtered_im > 0])
                        if clear_pixels > 0:
                            cloud_percentage = cloudy_pixels / clear_pixels
                        else:
                            cloud_percentage = 1

                        # Only consider 50% no clouds
                        if cloud_percentage <= 0.3:

                            # Now hide all cloud, background ...
                            filtered_im[filtered_im < -1] = np.nan

                            # Get mean, median ...
                            means = np.nanmean(filtered_im, axis=(1, 2))
                            medians = np.nanmedian(filtered_im, axis=(1, 2))
                            maxs = np.nanmax(filtered_im, axis=(1, 2))
                            mins = np.nanmin(filtered_im, axis=(1, 2))

                            # Filter indicator (= 0 usable, = 1 filtered)
                            control = 0

                            # Filtering values >1 or <0 in reflects
                            for band in [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]:

                                band_mean = means[band]
                                if band_mean > 1 or band_mean < 0 or np.isnan(band_mean):
                                    control = 1

                            if control != 1:

                                stds = np.nanstd(filtered_im, axis=(1, 2))
                                ndvi = np.nanmean((filtered_im[7] - filtered_im[3]) /
                                                  (filtered_im[7] + filtered_im[3]))

                                # Filtering values >1 or <0 in ndvi
                                if 1 > ndvi > 0:

                                    ndvi_std = np.nanstd((filtered_im[7] - filtered_im[3]) /
                                                         (filtered_im[7] + filtered_im[3]))

                                    # Filtering values >0.2
                                    if ndvi_std < 0.2:

                                        dvi = np.nanmean(filtered_im[7] - filtered_im[3])
                                        dvi_std = np.nanstd(filtered_im[7] - filtered_im[3])

                                        indexes = [ndvi, ndvi_std, dvi, dvi_std]
                                        crops = [current_polygon['croptype_2016'], current_polygon['croptype_2017'],
                                                 current_polygon['croptype_2018'], current_polygon['croptype_2019'],
                                                 current_polygon['croptype_2020'], current_polygon['croptype_2021'],
                                                 current_polygon['croptype_2022']]
                                        yields = [current_polygon['yield_2016'], current_polygon['yield_2017'],
                                                 current_polygon['yield_2018'], current_polygon['yield_2019'],
                                                 current_polygon['yield_2020'], current_polygon['yield_2021'],
                                                 current_polygon['yield_2022']]

                                        if region == '1':
                                            idd = parcel + 1
                                        if region == '2':
                                            idd = parcel + 106 + 1

                                        # Save current date data
                                        parcel_data.append([S2_im_path[-19:-11], current_polygon['lat'], current_polygon['long'], idd])
                                        parcel_crops.append(crops)
                                        parcel_yields.append(yields)
                                        parcel_means.append(means)
                                        parcel_stds.append(stds)
                                        parcel_indexes.append(indexes)
                                        parcel_mins.append(mins)
                                        parcel_maxs.append(maxs)
                                        parcel_medians.append(medians)
                                        parcel_geoms.append([multipolygon_reprojected, current_polygon['geometry']])

        # Save full parcel data
        pd_data = pd.DataFrame(parcel_data, columns=columns_data)
        pd_crops = pd.DataFrame(parcel_crops, columns=columns_crops)
        pd_yields = pd.DataFrame(parcel_yields, columns=columns_yields)
        pd_indexes = pd.DataFrame(parcel_indexes, columns=columns_indexes)
        pd_means = pd.DataFrame(parcel_means, columns=columns_means)
        pd_stds = pd.DataFrame(parcel_stds, columns=columns_stdev)
        pd_mins = pd.DataFrame(parcel_mins, columns=columns_mins)
        pd_maxs = pd.DataFrame(parcel_maxs, columns=columns_maxs)
        pd_medians = pd.DataFrame(parcel_medians, columns=columns_medians)
        pd_geoms = pd.DataFrame(parcel_geoms, columns=columns_geoms)

        final_csv = pd.concat([pd_data, pd_crops, pd_yields, pd_indexes, pd_means, pd_stds,
                               pd_mins, pd_maxs, pd_medians, pd_geoms], axis=1)

        # ORDER BY DATE
        final_csv = final_csv.sort_values(by=['date'])

        # Filter data series > 8
        if final_csv.shape[0] > 8:
            if region == '1':
                final_csv.to_csv(os.path.join(output_path, str(parcel + 1) + '.csv'), index=False)
            if region == '2':
                final_csv.to_csv(os.path.join(output_path, str(parcel + 1 + 106) + '.csv'), index=False)

            print( 'Parcel ' + str(idd) + ' done.')


