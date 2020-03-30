import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime

def load_data():
    data_dir = "../../../COVID-19/csse_covid_19_data/"
    covid_confirmed = pd.read_csv(data_dir+'csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    covid_deaths = pd.read_csv(data_dir+'csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    covid_recovered = pd.read_csv(data_dir+'csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

    return covid_confirmed, covid_deaths, covid_recovered


def logistic(t, a, b, c, d):
    return c + (d - c)/(1 + a * np.exp(- b * t))

def exponential(t, a, b, c):
    return a * np.exp(b * t) + c

def plotCasesandPredict(dataframe, column, country, days, mostrecentdate):

    co = dataframe[dataframe[column] == country].iloc[:,4:].T.sum(axis = 1)
    co = pd.DataFrame(co)
    co.columns = ['Cases']
    co = co.loc[co['Cases'] > 0]

    y = np.array(co['Cases'])
    x = np.arange(y.size)

    recentdbltime = float('NaN')

    if len(y) >= 7:

        current = y[-1]
        lastweek = y[-8]

        if current > lastweek:
            print('\n** Based on Most Recent Week of Data **\n')
            print('\tConfirmed cases on',co.index[-1],'\t',current)
            print('\tConfirmed cases on',co.index[-8],'\t',lastweek)
            ratio = current/lastweek
            print('\tRatio:',round(ratio,2))
            print('\tWeekly increase:',round( 100 * (ratio - 1), 1),'%')
            dailypercentchange = round( 100 * (pow(ratio, 1/7) - 1), 1)
            print('\tDaily increase:', dailypercentchange, '% per day')
            recentdbltime = round( 7 * np.log(2) / np.log(ratio), 1)
            print('\tDoubling Time (represents recent growth):',recentdbltime,'days')

    plt.figure(figsize=(10,5))
    plt.plot(x, y, 'ko', label="Observed infections")

    logisticworked = False
    exponentialworked = False

    try:
        lpopt, lpcov = curve_fit(logistic, x, y, maxfev=10000)
        lerror = np.sqrt(np.diag(lpcov))

        # for logistic curve at half maximum, slope = growth rate/2. so doubling time = ln(2) / (growth rate/2)
        ldoubletime = np.log(2)/(lpopt[1]/2)
        # standard error
        ldoubletimeerror = 1.96 * ldoubletime * np.abs(lerror[1]/lpopt[1])

        # calculate R^2
        residuals = y - logistic(x, *lpopt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        logisticr2 = 1 - (ss_res / ss_tot)

        if logisticr2 > 0.95:

            # show curve fit
            plt.plot(x, logistic(x, *lpopt), 'g--', label="Expected infections (logistic)")
            print('\n** Based on Logistic Fit**\n')
            print('\tR^2:', logisticr2)
            print('\tDoubling Time (during middle of growth): ', round(ldoubletime,2), '(±', round(ldoubletimeerror,2),') days')
            logisticworked = True

            # make predictions
            day_now = y.size-1 # what day we are on now
            future_day = day_now + days # how many days in the future we are predicting

            preds_log = [logistic(t,lpopt[0],lpopt[1],lpopt[2],lpopt[3]) for t in list(range(day_now,future_day,1))] # do pred
            x_log = list(range(day_now,future_day,1))
            y_log = preds_log
            plt.ticklabel_format(style = 'plain')
            plt.plot(x_log, y_log, 'y--', label="Predicted infections (logistic)") # plot predictions

            print('\n** Predicting day', future_day,'(',days,'days time)**\n')
            print('\tPredicted number of infections (logistic growth):',round(preds_log[-1]))


    except:
        pass

    try:
        epopt, epcov = curve_fit(exponential, x, y, bounds=([0,0,-100],[100,0.9,100]), maxfev=10000)
        eerror = np.sqrt(np.diag(epcov))

        # for exponential curve, slope = growth rate. so doubling time = ln(2) / growth rate
        edoubletime = np.log(2)/epopt[1]
        # standard error
        edoubletimeerror = 1.96 * edoubletime * np.abs(eerror[1]/epopt[1])

        # calculate R^2
        residuals = y - exponential(x, *epopt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        expr2 = 1 - (ss_res / ss_tot)

        if expr2 > 0.95:
            plt.plot(x, exponential(x, *epopt), 'r--', label="Expected infections (exponential)")
            print('\n** Based on Exponential Fit **\n')
            print('\tR^2:', expr2)
            print('\tDoubling Time (represents overall growth): ', round(edoubletime,2), '(±', round(edoubletimeerror,2),') days')
            exponentialworked = True

            # make predictions
            day_now = y.size-1 # what day we are on now
            future_day = day_now + days # how many days in the future we are predicting

            preds_exp = [exponential(t,epopt[0],epopt[1],epopt[2]) for t in list(range(day_now,future_day,1))] # do pred
            x_exp = list(range(day_now,future_day,1))
            y_exp = preds_exp
            plt.ticklabel_format(style = 'plain')
            plt.plot(x_exp, y_exp, 'b--', label="Predicted infections (exponential)") # plot predictions

            print('\n** Predicting day', future_day,'(',days,'days time)**\n')
            print('\tPredicted number of infections (exponential growth):',round(preds_exp[-1]))

    except:
        pass

    plt.title(country + ' Cumulative Confirmed COVID-19 Cases. (Updated on '+mostrecentdate+')', fontsize="x-large")
    plt.xlabel('Days', fontsize="x-large")
    plt.ylabel('Total Confirmed Cases', fontsize="x-large")
    plt.legend(fontsize="x-large")
    #plt.show()
    plt.yscale('linear')
    plt.savefig('../../data/figures/'+country+'_linear.png')
    #plt.yscale('log')
    #plt.savefig('../../data/figures/'+country+'_log.png')

    if logisticworked and exponentialworked:
        if round(logisticr2,2) > round(expr2,2):
            return [ldoubletime, ldoubletimeerror, recentdbltime,lpopt,round(preds_log[-1])]
        else:
            return [edoubletime, edoubletimeerror, recentdbltime,epopt,round(preds_exp[-1])]

    if logisticworked:
        return [ldoubletime, ldoubletimeerror, recentdbltime,lpopt,round(preds_log[-1])]

    if exponentialworked:
        return [edoubletime, edoubletimeerror, recentdbltime,epopt,round(preds_exp[-1])]
    else:
        return [float('NaN'), float('NaN'), recentdbltime,float('NaN'),float('NaN')]

def main(days):
    print(f'Start modeling COVID-19 cases based on latest data!')

    # load data
    covid_confirmed, covid_deaths, covid_recovered = load_data()

    dataframe = covid_confirmed
    dates = dataframe.keys()
    mostrecentdate = dates[-1]

    cases = dataframe.iloc[:,[1,-1]].groupby('Country/Region').sum().sort_values(by = mostrecentdate, ascending = False)
    topcountries = cases[cases[mostrecentdate] >= 100].index
    # countries = ['Italy','US','Switzerland','Iran','United Kingdom','Germany','Spain']

    column = "Country/Region"


    timestamp = datetime.now()

    results = {"results":[],"timestamp":timestamp}
    results_keys = ['country_code','country_name','resources_capacity','confirmed','deaths','recovered','confirmed_prediction_3w','deaths_prediction_3w','recovered_prediction_3w']

    for c in topcountries[:5]:
        # run
        dbltime,dbltimeerr,recentdbltime,params,pred = plotCasesandPredict(dataframe,column,c,days, mostrecentdate)

        # initialise dict of results
        results_dict = dict.fromkeys(results_keys)

        # generate dict values
        country_code = "TEST" # add later
        country_name = c
        resources_capacity = "TEST" # resources
        confirmed = int(dataframe[dataframe['Country/Region']==c].iloc[:,-1].sum())
        deaths = int(covid_deaths[covid_deaths['Country/Region']==c].iloc[:,-1].sum())
        recovered = int(covid_recovered[covid_recovered['Country/Region']==c].iloc[:,-1].sum())
        confirmed_prediction_3w = int(pred) # we take the prediction
        deaths_prediction_3w = int(deaths/confirmed * confirmed_prediction_3w) # use current perc deaths
        recovered_prediction_3w = int(recovered/confirmed * confirmed_prediction_3w) # use current perc recovered

        # populate dict
        results_dict['country_code'] = country_code
        results_dict['country_name'] = country_name
        results_dict['resources_capacity'] = resources_capacity
        results_dict['confirmed'] = confirmed
        results_dict['deaths'] = deaths
        results_dict['recovered'] = recovered
        results_dict['confirmed_prediction_3w'] = confirmed_prediction_3w
        results_dict['deaths_prediction_3w'] = deaths_prediction_3w
        results_dict['recovered_prediction_3w'] = recovered_prediction_3w

        # append to master dict
        results['results'].append(results_dict)

    return results


if __name__ == "__main__":
    days = 21 # 3 weeks prediction
    results = main(days)
    print(results)