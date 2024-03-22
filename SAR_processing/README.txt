Author: Lucio Mascolo, Univeristy of Valencia (UV)
####################################################################################################
#         STEP-1: SUBSET THE "RAW" GRD DATA WITH THE POLYGON OF THE AREA OF INTEREST               #
####################################################################################################

This is done with the code "subset-Sigma0_from-GRD.py" which uses the SNAP graph "Subset_GRD.xml"

The GRD processing steps are:
- Apply-Orbit-File
- ThermalNoiseRemoval
- Calibration
- Subset
- Speckle-Filter (3x3 boxcar)
- Terrain-Correction
The resulting output image (containing Sigma0_VV and Sigma0_VH) is in GeoTIFF

# Aux info:
The folder "raw_GRD_folder" contains the time series of GRD data to process
The folder "processed_Sigma0" contains the processed products
The folder "graphs" contains the SNAP graphs called in the code
The folder "ROI_area" contains the ROI of the area of interest

################################################################################################################
# STEP-2: EXSTRACT Sigma0_VV, Sigma0_VH, and "CHILDREN" PARAMETERS FROM THE POLYGONS OF THE CROPS OF INTEREST  #
################################################################################################################

This is done with the code "extract_Sigma0_from_polygons.py" which uses the Sigma0_VV and Sigma0_VH products 
derived at the previous step.

The set of SAR parameters, extracted for each polygon, is:
- Sigma0_VV and Sigma0_VH (dB scale)
- the VH/VV ratio =  Sigma0_VH/Sigma0_VV (dB scale)
- the Radar Vegetation Index (RVI) computed as RVI = 4*Sigma0_VH/(Sigma0_VH+Sigma0_VV) (linear scale)

For each parameter, the mean value is computed over the polygons pixels for each GRD image of the time series
These results are saved in a GeoPanda dataframes, which include: the date of the Sentinel-1 acquisitions; the ID of the polygons, and the corresponding yield
################################################################################################################






