#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:24:17 2024

@author: mascolo
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')

import os
import numpy as np
import spectral as sp
from glob import glob
import json
import geojson
import fiona
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from datetime import datetime

## Author: Lucio Mascolo, Univeristy of Valencia (UV)

############################ SET INPUT VARIABLES ##############################  
crop_type       = 'wheat'
season          = '2017'      ## season
start_yyyymmdd  = '20161101'  ## time series starting date: string 'yyyymmdd'
end_yyyymmdd    = '20170415'  ## time series ending date: string 'yyyymmdd'

## Folder containing the processed GRD_HD images
sar_data_folder = '/media/ucg_ag/Sentinel1/GRD_HD/Processed/TANTA/ORB-167/'

## Folder containing the fields polygons
poly_folder     = '/home/mascolo/Desktop/DATA/Ground-data/EO_africa/polygons/'
poly_fname      = 'polys_buffered_data.gpkg'

## Output folder
out_folder      = '/home/mascolo/Desktop/DATA/dataframes_EO_africa/'
###############################################################################

###############################################################################
##                          LOAD FIELDS POLYGONS                             ##
###############################################################################

## Load ALL the polygons of the fields  with geopanda -----------------------##
allpolys_df     = gpd.read_file(poly_folder+poly_fname)
nall            = len(allpolys_df)                ## number of all the polygons
# multy_coords   = np.array((npoly,),dtype=object)
EPSG_code       = allpolys_df.crs.srs

## Select only the season of interest in the original dataframe
season_polys    = {
                    'fid'             : allpolys_df .index+1,
                    'lat'             : allpolys_df['lat'],
                    'long'            : allpolys_df['long'],
                    'croptype_'+season: allpolys_df['croptype_'+season],
                    'yield_'+season   : allpolys_df['yield_'+season],
                    'geometry'        : allpolys_df['geometry']
                   }

season_polys    = gpd.GeoDataFrame( pd.DataFrame(season_polys), crs=EPSG_code ) ## convert to gpd dataframe
season_polys    = season_polys.dropna() ## drop the nans

## Select columns which include the crop of interest (wheat) -> still not the definitive for 'wheat - rice'
polys_crop      = season_polys.loc[season_polys['croptype_'+season].str.contains(crop_type)]
ncrops          = len(polys_crop)
## polys_crop is not the definitve list of wheat fields
  
for index, crops, cropyield in zip(polys_crop.index,polys_crop['croptype_'+season],polys_crop['yield_'+season]):

    crop_split  = crops.split('-')
    yield_split = cropyield.split('-')

    whereiscrop = [ ii for ii in range(len(crop_split)) if crop_type in crop_split[ii]][0]

    polys_crop['croptype_'+season][index] = crop_split[whereiscrop]
    polys_crop['yield_'+season][index]    = yield_split[whereiscrop]
###############################################################################

###############################################################################
##                            LIST OF SAR IMAGES                             ##
###############################################################################
s1_files      = sorted(glob(os.path.join(sar_data_folder,'*.tif'))) ## list of S1 files
s1_files      = [s1_files[idx0].split(sar_data_folder)[1] for idx0 in range(len(s1_files))] ## S1 filename
s1_dates      = [s1_files[idx0][17:25] for idx0 in range(len(s1_files)) ] ## S1 dates (strings)
s1_files      = [s1_files[np.argsort(s1_dates)[idx0]] for idx0 in range(len(s1_files))] ## sort the files according to the dates
s1_dates      = sorted(s1_dates) ## sort the date
s1_dates      = [datetime.strptime(s1_dates[idx0],'%Y%m%d').date() for idx0 in range(len(s1_files))] ## transform dates in datetime

## Subset the images by end/starting date
start_date    = datetime.strptime(start_yyyymmdd,'%Y%m%d').date() 
end_date      = datetime.strptime(end_yyyymmdd,'%Y%m%d').date() 
images_season = [s1_files[idx0] for idx0 in range(len(s1_files)) if ((start_date <= s1_dates[idx0]) & (s1_dates[idx0] <= end_date))]
dates_season  = [s1_dates[idx0] for idx0 in range(len(s1_files)) if ((start_date <= s1_dates[idx0]) & (s1_dates[idx0] <= end_date))]
dates_season  = pd.to_datetime(dates_season,format='%Y-%m-%d')
nimgs         = len(images_season)
###############################################################################


data_dic              = {}
data_dic['Season']    = []
data_dic['date']      = []
data_dic['fid']       = []
data_dic['sigma0_VH'] = []
data_dic['sigma0_VV'] = []
data_dic['VH/VV']     = []
data_dic['RVI']       = []
data_dic['yield']     = []
data_dic['geometry']  = [] 

data_df               = gpd.GeoDataFrame(pd.DataFrame(data_dic),crs=32636)



for n in range(nimgs):
    
    with rasterio.open(sar_data_folder+ images_season[n]) as img:
        
        out_meta_poly                    = img.meta.copy()
        img_epsg_code                    = out_meta_poly['crs'].to_epsg()
        polys_crop                       = polys_crop.to_crs(img_epsg_code)
        
        #ipolys=0
        for ipolys in range(ncrops):
                
            index       = polys_crop.index[ipolys]
    
            coordinates = [json.loads(polys_crop.to_json())['features'][ipolys]['geometry']]
            # #fld_id      = [json.loads(polys_crop.to_json())['features'][ipolys]['fid']]
            
            out_img_poly, out_transform_poly = mask(img,coordinates,crop=True)
            
            vh_power         = out_img_poly[0]
            vv_power         = out_img_poly[1]
            idx_valid_pixels = np.where(vh_power>0)
            
            
            #plt.imshow(10*np.log10(vh_power),cmap='gray')
     
            
            if len(idx_valid_pixels[0])>0:
                
                vhvv_ratio       = vh_power[idx_valid_pixels]/vv_power[idx_valid_pixels]
                rvi              = 4*vh_power[idx_valid_pixels]/(  vh_power[idx_valid_pixels]+vv_power[idx_valid_pixels]   )
                
                ## - Update all
                curr_dic = {
                    
                    'sigma0_VH' : np.mean( 10*np.log10(vh_power[idx_valid_pixels]) ),
                    'sigma0_VV' : np.mean( 10*np.log10(vv_power[idx_valid_pixels])),
                    'VH/VV'     : np.mean( 10*np.log10(vhvv_ratio)),
                    'RVI'       : np.mean(rvi),
                    'Season'    : season,
                    'date'      : dates_season[n],
                    'fid'       : polys_crop['fid'][index],
                    'yield'     : polys_crop['yield_'+season][index],
                    'geometry'  : polys_crop['geometry'][index],
                    }
                
                df_toappend      = pd.DataFrame(curr_dic,index=[0])
                gdf_toappend     = gpd.GeoDataFrame(df_toappend,crs=32636)   ## convert to geodataframe
                data_df          = pd.concat([data_df,gdf_toappend ], ignore_index=True) ## append to the geodataframe
          


###############################################################################
##                          SAVE IN THE OUTPUT FOLDER                        ##
###############################################################################       
fname_out = 'GRD-Yield_dataframe-'+crop_type+'-'+season+'.geojson'
path_out  = out_folder+fname_out
data_df.to_file(path_out, driver = 'GeoJSON')
###############################################################################














