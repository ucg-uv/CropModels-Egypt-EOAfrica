import subprocess



# Download ERA5 given start and end date
subprocess.run(['python', 'download_era5.py'])

# Extract S2 band values given a S2 database and geometries with reference yield and croptype data
subprocess.run(['python', 'extract_reflect.py'])

# Merge SAR data from previous codes
subprocess.run(['python', 'SAR_merge.py'])

# Calculates GDD and aggregates with previous reflectivity values
subprocess.run(['python', 'GDD.py'])

# Same as last code but use in case of SAR fusion model
subprocess.run(['python', 'GDD_SAR.py'])

# Filters, smoother and interpolates reflectivity values
subprocess.run(['python', 'filter.py'])

# Same as last code but use in case of SAR fusion model
subprocess.run(['python', 'SAR_filter.py'])

# Plot parcels by groups of yield with GDD
subprocess.run(['python', 'plot_yieldgroups.py'])

# Plot parcels by groups of yield with DOY
subprocess.run(['python', 'plot_yieldgroups_DOY.py'])

# Assessment of the different models with better coefficients
subprocess.run(['python', 'model_optic_wheat.py'])
subprocess.run(['python', 'model_optic_rice.py'])
subprocess.run(['python', 'model_fusion.py'])