# CropModels-Egypt-EOAfrica
Models for Crop type mapping and yield retrieval for wheat and rice in Nile Delta. This repository contains a series of Python scripts designed to classify crops and retrieve yield information using remote sensing data. The process involves downloading ERA5 climate data, extracting Sentinel-2 (S2) band values, merging Synthetic Aperture Radar (SAR) data, calculating Growing Degree Days (GDD), filtering, and smoothing reflectivity values, and assessing different models for optimal performance.

## Prerequisites

Before running these scripts, ensure you have the following installed:
- [Anaconda](https://www.anaconda.com/download)/[Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Python 3.11](https://docs.python.org/es/3.11/). (Included in conda)

## Installation

Then, you can run the following commands
Clone this repository to your local machine:
```
conda env create -f teledeteccion311.yml
conda activate teledeteccion311
pip install -r requirements.txt
```
## Usage

The process is divided into multiple steps, each performed by a separate script. Follow these steps in order to classify crops and retrieve yield information:
All the yield retrieval using optic and fusion (+SAR) data can be summarized in the code `main.py`.
Also, every step from `main.py` can be executed by step. The steps are the following:

1. **Download ERA5 Climate Data**
```
python download_era5.py
```
2. Replace `<start-date>` and `<end-date>` with the desired date range for the ERA5 climate data download.

2. **Extract Sentinel-2 Band Values**
Ensure you have a Sentinel-2 (S2) database and geometries with reference yield and crop type data available.
```
python extract_reflect.py
```
3. **Merge SAR Data**
```
python SAR_merge.py
```
4. **Calculate GDD and Aggregate Reflectivity Values**
```
python GDD.py
```
5. **GDD Calculation for SAR Fusion Model**
```
python GDD_SAR.py
```
6. **Filter, Smooth, and Interpolate Reflectivity Values**
```
python filter.py
```
7. **Filtering for SAR Fusion Model**
```
python SAR_filter.py
```
8. **Plot Parcels by Yield Groups with GDD**
 ```
python plot_yieldgroups.py
```
9. **Plot Parcels by Yield Groups with Day of Year (DOY)**
```
python plot_yieldgroups_DOY.py
```
10. **Assess Different Models**
 For wheat:
 ```
 python model_optic_wheat.py
 ```
 For rice:
 ```
 python model_optic_rice.py
 ```
 For SAR fusion model:
 ```
 python model_fusion.py
 ```

## Support

If you encounter any issues or have questions regarding the process, please open an issue in the repository.


## Additional Notes

- Ensure all data paths and dependencies are correctly set up in your environment before running the scripts.

Thank you for using this crop classification and yield retrieval process. We hope it assists you in your research and data analysis efforts.


