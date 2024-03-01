
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
from scipy.signal import savgol_filter
import numpy as np
import os
import pandas as pd
from datetime import datetime
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import geopandas as gpd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns

from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
main_path = os.path.getcwd()


optic_path = os.path.join(main_path,'OUTPUTS/filtered_100')
optic_list = os.listdir(optic_path)
SAR_path = os.path.join(main_path,'OUTPUTS/filtered_SAR')
SAR_list = os.listdir(SAR_path)

output_path = os.path.join(main_path,'OUTPUTS/plots_100')
fig, ax = plt.subplots(figsize=(9, 6))

fig_width = fig.get_figwidth()
fig_height = fig.get_figheight()
incre = 0
Y_boxplot = []
X_boxplot = []

all_X = []
all_Y = []

for year in range(2017, 2023):


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

    all_calculated_yield = []
    all_reference_yield = []
    all_current_gdd = []

    for parcel in range(1, 150):

        calculated_yield = []
        reference_yield = []
        current_gdd = []

        if (str(parcel) + '.csv' in optic_list) and (str(parcel) + '.csv' in SAR_list):

            current_optic_all = pd.read_csv(os.path.join(optic_path, str(parcel) + '.csv'))
            current_SAR_all = pd.read_csv(os.path.join(SAR_path, str(parcel) + '.csv'))

            current_optic_1 = current_optic_all[current_optic_all['SEASON'] == year]
            current_optic = current_optic_1[current_optic_1['CROPTYPE'] == 'wheat']
            current_SAR = current_SAR_all[current_SAR_all['SEASON'] == year]

            B11 = current_optic['B11'].values
            VHVV = current_SAR['VH/VV'].values

            if (len(current_optic) > 0) and (len(current_SAR) > 0):

                for gdd in range(0, len(current_optic.GDD)):
                    calculated_yield.append(-35504.528306818*B11[gdd] + 275.508825515759*(VHVV[gdd]) + 10710.0552405678)
                    reference_yield.append(np.round((current_optic['YIELD'].iloc[0]/0.42), 2))
                    current_gdd.append(current_optic['GDD'].iloc[gdd])

        if len(calculated_yield) > 0:
            all_calculated_yield.append(calculated_yield)
            all_reference_yield.append(reference_yield)
            all_current_gdd.append(current_gdd)

    if len(all_calculated_yield) > 0:
        incre = incre + 400

        all_r2 = []
        all_rmse = []
        all_gdd = []

        for gdd_day in range(0, len(all_current_gdd[0])):
            current_calculated_yield = []
            current_reference_yield = []
            for parcel_num in range(0, len(all_calculated_yield)):
                current_calculated_yield.append(all_calculated_yield[parcel_num][gdd_day])
                current_reference_yield.append(all_reference_yield[parcel_num][gdd_day])
            # Perform linear regression
            r2 = r2_score(current_reference_yield, current_calculated_yield)
            rmse = np.sqrt(mean_squared_error(current_reference_yield, current_calculated_yield))

            all_r2.append(r2)
            all_rmse.append(rmse)
            all_gdd.append(gdd_day)

        # Get less rmse day
        # Combine lists using zip
        combined_lists = list(zip(all_rmse, all_r2, all_gdd))
        # Sort based on values in all_rmse (index 0)
        sorted_combined_lists = sorted(combined_lists, key=lambda x: x[0])
        # Separate the sorted lists
        sorted_all_rmse, sorted_all_r2, sorted_all_gdd = zip(*sorted_combined_lists)
        stop = 0
        for value in range(0, len(sorted_all_rmse)):
            if stop == 0:
                gdd_index = sorted_all_gdd[value]
                if 500 <= all_current_gdd[0][gdd_index] <= 3200:
                #if True:
                    champion_rmse = sorted_all_rmse[value]
                    champion_r2 = sorted_all_r2[value]
                    champion_gdd = sorted_all_gdd[value]
                    stop = 1


        X = []
        Y = []
        Z = []

        for parcel_num in range(0, len(all_calculated_yield)):
            X.append(all_reference_yield[parcel_num][champion_gdd])
            Y.append(all_calculated_yield[parcel_num][champion_gdd])
            Z.append(all_current_gdd[0][champion_gdd])

        # Round Y values to 2 decimals
        Y_rounded = [round(y, 2) for y in Y]
        X = [int(x) for x in X]

        # Order
        combined_lists = list(zip(X, Y_rounded))
        sorted_combined_lists = sorted(combined_lists, key=lambda x: x[0])
        X, Y_rounded = zip(*sorted_combined_lists)

        all_X.append(X)
        all_Y.append(Y_rounded)

        # Create a DataFrame for Seaborn
        data = {'X Values': X, 'Y Values': Y_rounded}
        df = pd.DataFrame(data)

        # Set boxplot parameters
        boxplot_params = {
            'width': 0.3,
            'boxprops': dict(facecolor='white', edgecolor=color)}

        #plt.boxplot(Y_rounded, labels=['6904', '7142', '7380', '7619','7857'])
        sns.boxplot(x='X Values', y='Y Values', data=df, showfliers=False, **boxplot_params, order=X)
        sns.stripplot(x='X Values', y='Y Values', data=df, color=color, size=5, jitter=False)
        #ax.text(4.6, 7250-incre, str(year) + ' (' + str(len(X)) + ')\n' + 'RMSE = ' + str(int(champion_rmse)), color=color, fontsize=10, ha='left',
        #         va='center')
        ax.text(4.6, 7750-incre, str(year) + ' (' + str(len(X)) + ')\n' + 'RMSE = ' + str(int(champion_rmse))
                + '\n GDD = ' + str(int(Z[0])), color=color, fontsize=10, ha='left', va='center')


# Set labels for axes
plt.xlabel('Reference yield (kg/ha)')
#plt.xticks(['7857', '7380', '7619', '6904', '7142'])
#plt.xticks(['6904', '7142', '7380', '7619','7857'])
plt.ylabel('Calculated yield (kg/ha)')

# Set the title of the plot
plt.title('CROP: wheat - All Seasons - S1 + S2')

# Show the plot
plt.tight_layout()
#plt.show()
#pass
#plt.savefig(os.path.join(output_path, 'all_fusion.png'), bbox_inches='tight')


means_1_all = []
means_2_all = []
means_3_all = []
means_4_all = []
means_5_all = []

for season in range(0, len(all_Y)):
    group_1 = []
    group_2 = []
    group_3 = []
    group_4 = []
    group_5 = []
    current_season = all_Y[season]
    current_season_ref = all_X[season]
    for value in range(0, len(current_season)):
        if current_season_ref[value] == 6904:
            group_1.append(current_season[value])
        if current_season_ref[value] == 7142:
            group_2.append(current_season[value])
        if current_season_ref[value] == 7380:
            group_3.append(current_season[value])
        if current_season_ref[value] == 7619:
            group_4.append(current_season[value])
        if current_season_ref[value] == 7857:
            group_5.append(current_season[value])

    if len(group_1) > 0:
        mean_1 = np.mean(group_1)
        means_1_all.append(mean_1)
    if len(group_2) > 0:
        mean_2 = np.mean(group_2)
        means_2_all.append(mean_2)
    if len(group_3) > 0:
        mean_3 = np.mean(group_3)
        means_3_all.append(mean_3)
    if len(group_4) > 0:
        mean_4 = np.mean(group_4)
        means_4_all.append(mean_4)
    if len(group_5) > 0:
        mean_5 = np.mean(group_5)
        means_5_all.append(mean_5)

calc_data_pack = [means_1_all, means_2_all, means_3_all, means_4_all, means_5_all]
calc_data = [item for sublist in calc_data_pack for item in sublist]
ref_data = [6904, 7142,7142, 7380,7380, 7619,7619,7619, 7857]

r2 = r2_score(ref_data, calc_data)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(ref_data, calc_data))


means_means_1_all = np.nanmean(means_1_all)
means_means_2_all = np.nanmean(means_2_all)
means_means_3_all = np.nanmean(means_3_all)
means_means_4_all = np.nanmean(means_4_all)
means_means_5_all = np.nanmean(means_5_all)

calc_data = [means_means_1_all, means_means_2_all, means_means_3_all, means_means_4_all, means_means_5_all]
ref_data = [6904, 7142, 7380,7619, 7857]

r2 = r2_score(ref_data, calc_data)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(ref_data, calc_data))

pass
'''
    # Scatter
    
    if len(X) > 1:

        X = np.array(X)
        Y = np.array(Y)

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(X, Y)

        # Create scatter plot
        plt.figure()
        plt.scatter(X, Y)

        # Add trend line
        line = slope * X + intercept
        plt.plot(X, line, color='red')

        # Add equation, R2 value, title, and axis labels
        equation = f'Y = {slope:.2f}X + {intercept:.2f}'
        r_squared = f'R2 = {r_value ** 2:.2f}'
        rmse_text = f'RMSE = {champion_rmse:.2f} kg/ha'
        # Equation and R2 can be placed in (x, y) coordinates just removing transform=plt.gca().transAxes
        plt.text(0.05, 0.9, equation, fontsize=10, color='blue', transform=plt.gca().transAxes)
        plt.text(0.05, 0.85, r_squared, fontsize=10, color='blue', transform=plt.gca().transAxes)
        plt.text(0.05, 0.80, rmse_text, fontsize=10, color='blue', transform=plt.gca().transAxes)

        plt.title('Season: ' + str(year) + ' - GDD: ' + str(int(Z[0])))
        plt.xlabel('Reference yield (kg/ha)')
        plt.ylabel('Calculated yield (kg/ha)')

        plt.ylim(6000, 8000)
        plt.xlim(6000, 8000)
        # Save the figure
        plt.savefig(os.path.join(output_path, str(year) + '.png'), bbox_inches='tight')
'''
