#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:31:11 2024

@author: mascolo
"""

#                       IMPORT MODULES                      #
import os
import sys
import shutil
import fiona
import json
import numpy as np
import rasterio as rio
from osgeo import gdal

from shapely.geometry import Polygon
from pathlib import Path
from logging import raiseExceptions
from glob import glob
from rich import print
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
# from sidra_toolbox import dprvi
# from sidra_toolbox import resampleRasters

import time

## Author: Lucio Mascolo, Univeristy of Valencia (UV)

###############################################################################
##                                SET INPUTS                                 ##

input_grd_folder     = '/media/ucg_ag/Sentinel1/EO_africa_DoYorself/raw_GRD_folder/'

input_roi_folder     = '/media/ucg_ag/Sentinel1/EO_africa_DoYorself/ROI_area/' 
roi_name             = 'TANTA.geojson'

snap_graphs_folder   = '/media/ucg_ag/Sentinel1/EO_africa_DoYorself/graphs/'

output_sigma0_folder = '/media/ucg_ag/Sentinel1/EO_africa_DoYorself/processed_Sigma0/'

###############################################################################
##                                 EXECUTE                                   ##
###############################################################################

## Load the GRD files to process
grd_files   = sorted(glob(os.path.join(input_grd_folder,'*.zip'))) ## list of S1 files

## Load the ROI of the area of interest
roi         = os.path.join(input_roi_folder,roi_name)

## Extract coordinates of the ROI polygon
with open(roi) as f:
        gjson = json.load(f)
        coords = np.array(gjson['features'][0]['geometry']['coordinates']).squeeze() # Parse coordinates

long_min = np.min(coords[:,0])
long_max = np.max(coords[:,0])
lat_min  = np.min(coords[:,1])
lat_max  = np.max(coords[:,1])

## Get EPSG code based on lat, long coordinates of ROI polygon
utm_crs_list = query_utm_crs_info(
    datum_name="WGS 84",
    area_of_interest=AreaOfInterest(
        west_lon_degree=long_min,
        south_lat_degree=lat_min,
        east_lon_degree=long_max,
        north_lat_degree=lat_max,
    ),
)
roi_wkt = Polygon([[long_min, lat_min], [long_max, lat_min], [long_max, lat_max], [long_min, lat_max], [long_min, lat_min]]).wkt

## Turn Python variables into Shell variables
os.environ['roi']      = roi_wkt
os.environ['dem_name'] =  'Copernicus 30m Global DEM' ## DEM to use in SNAP for the geocoding
os.environ['crs']      = 'EPSG:' + utm_crs_list[0].code


###############################################################################
##                          RUN THE PROCESSING                               ##
###############################################################################
start         = time.time()
for grd_file in grd_files:
    
    input_filename            = Path(grd_file).stem
    output_filename           = output_sigma0_folder + input_filename + '_Sigma0.tif'
    os.environ['input_file']  = grd_file
    os.environ['output_file'] = output_filename
    
    if not os.path.exists(output_filename):
            print(f'[bold deep_sky_blue2] Generating and subsetting Sigma0 GRD image from {input_filename}... [/bold deep_sky_blue2]')
            
            # It is mandatory for $dem and $roi to be between "". If this is not the case bash will not be able to correctly read them
            os.system('/media/nas/snap/bin/gpt '+snap_graphs_folder+'Subset_GRD.xml ' + '-Pinput=$input_file -Poutput=$output_file -Pcrs=$crs -Proi_wkt="$roi" -Pdem="$dem_name"')

    else:
            print(f'[bold dark_orange] WARNING: Image {input_filename} already processed. Skiping to next image...[/bold dark_orange]')
        
    print('[bold chartreuse3] Done! [/bold chartreuse3]')
print((time.time()-start))



