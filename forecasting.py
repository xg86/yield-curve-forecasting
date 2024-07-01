# USER INPUT HERE
import datetime
start_date = datetime.datetime(2011, 11,  1) # starting date for the training window, first available date 2011/11/01
end_date   = datetime.datetime(2018, 5, 30) # end date for the training windows, last available date 2019/10/31
forecast_step = 30 # how many *day* to forecast forward

# IMPORT PACKAGES
import numpy as np
import numpy.matlib
import math
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from mpl_toolkits import mplot3d

# import our custom functions
from yield_curve_functions import DNS_OLS, DNS_formula, forecast_RW_fct, params_VAR
from yield_curve_functions import forecast_DNS_VAR, forecast_DNS_VAR, forecast_VAR, forecast_AR, forecast_DNS_KF
from yield_curve_functions import forecast_DNS_VAR_yw, forecast_DNS_VAR_yw, forecast_VAR_yw, forecast_AR_yw, forecast_DNS_KF_explosivcor

# IMPORT DATA
#y_df = pd.read_excel('US_daily.xlsx', columns = ['dates',1/12,3/12,6/12,1,2,3,5,7,10,20,30], index='dates');
#y = y_df.to_numpy()
#matu = np.array([[1/12,3/12,6/12,1,2,3,5,7,10,20,30]])

# subset of the data based on user's input
#y = y[ (y[:,0] >= start_date) & (y[:,0] <= end_date) ,:]

# IMPORT DATA
y_full_df = pd.read_excel('US_daily.xlsx', names=['dates', 1 / 12, 3 / 12, 6 / 12, 1, 2, 3, 5, 7, 10]);
y_full_df.set_index('dates')
y_full = y_full_df.to_numpy()
matu = np.array([[1 / 12, 3 / 12, 6 / 12, 1, 2, 3, 5, 7, 10]])

dates = y_full[:, 0]

current = 1  # this variable keeps track of the dates in the original dataset that have already been added. It is a row index in the original table.
currentDate = np.datetime64(dates[0])  # this variable keeps track of all dates that need to be added.

# The following two tables will be concatenated horizontally to create the full, new dataset
CompleteTable = np.array(
    [y_full[0, 1:]])  # Table with added yields (has copied lines where extra dates have been added)
CompleteDates = np.array([[currentDate]], dtype='datetime64')  # Will be the full dates column

AddDay = np.timedelta64(1, 'D')

cdnp = np.array([[currentDate]],
                dtype='datetime64')  # single entry array. Used to have a compatible format (np.array) for adding the dates to CompleteDates.

while current < y_full.shape[0]:
    currentDate = currentDate + AddDay
    cdnp[0][0] = currentDate
    CompleteDates = np.hstack((CompleteDates, cdnp))
    dateInTable = np.datetime64(dates[current])

    if dateInTable != currentDate:
        CompleteTable = np.vstack((CompleteTable, CompleteTable[-1]))  # copies last available line into the table

    if dateInTable == currentDate:
        CompleteTable = np.vstack((CompleteTable, y_full[current, 1:]))  # adds yield curve corresponding to currentDate
        current = current + 1

# Updating to full table
y_full = np.hstack((CompleteDates.transpose(), CompleteTable))

# subset of the data based on user's input
y = y_full[(y_full[:, 0] >= start_date) & (y_full[:, 0] <= end_date), :]

# find the forecast date
forecast_date = end_date + timedelta(days=forecast_step)
y_forecast = y_full[y_full[:, 0] == forecast_date, 1:]
if y_forecast.size == 0:
    print("Error: the given forecast step fall outside of the dataset")

# find the b4_forecast date
b4_forecast_date = end_date - timedelta(days=forecast_step)
y_b4_forecast = y_full[y_full[:, 0] == b4_forecast_date, 1:]
if y_b4_forecast.size == 0:
    print("Error: the given forecast step fall outside of the dataset")


# seperating dates and yields
dates = np.array([y[:, 0]])
y = np.delete(y, 0, 1)
y = np.array(y, dtype=float)

#plot_x = matu
# plot_y = [x.year for x in dates[0,:]]

def plot_start_date(start_date):
    decimal_start = start_date.year
    index_in_year = start_date.day
    month = start_date.month
    if (month >= 2):
        index_in_year += 31
    if (month >= 3):
        index_in_year += 28
    if (month >= 4):
        index_in_year += 31
    if (month >= 5):
        index_in_year += 30
    if (month >= 6):
        index_in_year += 31
    if (month >= 7):
        index_in_year += 30
    if (month >= 8):
        index_in_year += 31
    if (month >= 9):
        index_in_year += 31
    if (month >= 10):
        index_in_year += 30
    if (month >= 11):
        index_in_year += 31
    if (month >= 12):
        index_in_year += 30

    decimal_start += index_in_year / 365
    return decimal_start


#plot_y = (np.arange((dates[0, :]).shape[0]) / 365) + plot_start_date(start_date)

#plot_x, plot_y = np.meshgrid(plot_x, plot_y)
#plot_z = y

#fig = plt.figure(figsize=(15, 15))
#ax = plt.axes(projection='3d')

#ax.plot_surface(plot_x, plot_y, plot_z, cmap='viridis', edgecolor='none')
#ax.set_title('US treasury yields in the selected window')
#ax.set_ylabel('')
#show surface
#plt.show()


# OLS fitting of the coefficients
our_lambda = 0.496 #result of commented computation
ts = DNS_OLS(y,matu,our_lambda)
tsf = pd.DataFrame(ts) #some functions require the data in Pandas instead of Numpy

data = y
dataf = pd.DataFrame(y)

p = matu.shape[1] #p = nb maturities
nb_dates = y.shape[0]

# PRODUCE FORECASTS
forecast_RW = forecast_RW_fct(data,forecast_step) #Random walk
step_two_VAR = forecast_DNS_VAR(tsf,forecast_step) #Two Step DNS with VAR(1)
step_two_VAR_yw = forecast_DNS_VAR_yw(ts,forecast_step) #Two Step DNS with VAR(1), method of moments estimator
forecast_VARn = forecast_VAR(dataf,forecast_step)
forecast_VARn_yw = forecast_VAR_yw(data,forecast_step)
forecast_ARn = forecast_AR(data,forecast_step)
forecast_ARn_yw = forecast_AR_yw(data,forecast_step)

#We choose to initialise the EM algorithm with VAR fitted parameters, because they are likely to make it converge to the global maximum.
state_init = ts[0]
params_init = params_VAR(tsf)
offset_init = params_init[0,:]
transition_init = np.array(params_init[1:,:]).transpose()

forecast_KF = forecast_DNS_KF(dataf,matu,our_lambda,state_init,offset_init,transition_init,forecast_step) #One Step DNS with VAR(1).
forecast_KF_explosivcor = forecast_DNS_KF_explosivcor(dataf,matu,our_lambda,state_init,offset_init,transition_init,forecast_step) #One Step DNS with VAR(1).


tau_grid = np.linspace(start=0.001, stop=10, num=100)

fig = plt.figure(figsize=(15,8))
#plt.plot( tau_grid, DNS_formula( tau_grid, step_two_VAR[-1,:] , our_lambda), 'b--' )
plt.plot( tau_grid, DNS_formula( tau_grid, step_two_VAR_yw[-1,:] , our_lambda), 'b-' )
plt.plot( tau_grid, DNS_formula( tau_grid, forecast_KF[-1,:], our_lambda ), 'r--'  )
plt.plot( tau_grid, DNS_formula( tau_grid, forecast_KF_explosivcor[-1,:], our_lambda ), 'r-'  )
#plt.plot( matu.flatten(), forecast_ARn[-1,:], color='grey', marker='o', ls=':' )
plt.plot( matu.flatten(), forecast_ARn_yw[-1,:], color='grey', marker='o', ls='--' )
#plt.plot( matu.flatten(), forecast_VARn[-1,:], color='orange', marker='o', ls=':' )
plt.plot( matu.flatten(), forecast_VARn_yw[-1,:], color='orange', marker='o', ls='--' )
plt.plot( matu.flatten(), forecast_RW[-1,:], color='green', marker='o', ls='-.' )

plt.plot( matu.flatten(), y_forecast.flatten() ,'k-o')
plt.plot( matu.flatten(), y_b4_forecast.flatten(), 'm-o')
plt.title('forcast <'+ str(forecast_step) +' days > US treasury yields in selected window')
plt.legend(['DNS 2-step',
            'DNS 1-step',
            'DNS 1-step (explosivity correction)',
            'AR',
            'VAR',
            'RW',
            'truth', 'b4-truth'])
plt.xlabel('maturity [years]')
plt.ylabel('yield [percents]')
plt.show()