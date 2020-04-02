

import csv
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime

import fetch_hopkins
import populate_results


def get_country_code():
    countries = { 'Diamond Princess':'DP',
                  'Aruba':'AW', 
                  'Afghanistan':'AF', 
                  'Angola':'AO', 
                  'Anguilla':'AI', 
                  'Aland Islands':'AX', 
                  'Albania':'AL',
                  'Andorra':'AD',
                  'United Arab Emirates':'AE',
                  'Argentina':'AR',
                  'Armenia':'AM',
                  'American Samoa':'AS',
                  'Antarctica':'AQ',
                  'French Southern Territories':'TF',
                  'Antigua and Barbuda':'AG',
                  'Australia':'AU',
                  'Austria':'AT',
                  'Azerbaijan':'AZ',
                  'Burundi':'BI',
                  'Belgium':'BE',
                  'Benin':'BJ',
                  'Bonaire, Sint Eustatius and Saba':'BQ',
                  'Burkina Faso':'BF',  'Bangladesh':'BD',
                  'Bulgaria':'BG',
                  'Bahrain':'BH',
                  'Bahamas':'BS',
                  'Bosnia and Herzegovina':'BA',
                  'Saint Barthélemy':'BL', #non-ascii
                  'Belarus':'BY',
                  'Belize':'BZ',
                  'Bermuda':'BM',
                  'Bolivia, Plurinational State of':'BO',
                  'Brazil':'BR',
                  'Barbados':'BB',
                  'Brunei':'BN',
                  'Bhutan':'BT',
                  'Bouvet Island':'BV',
                  'Botswana':'BW',
                  'Central African Republic':'CF',
                  'Canada':'CA',
                  'Cocos (Keeling) Islands':'CC',
                  'Switzerland':'CH',
                  'Chile':'CL',
                  'China':'CN',
                  'Cote d\'Ivoire':'CI',
                  'Cameroon':'CM',
                  'Congo, The Democratic Republic of the':'CD',
                  'Congo':'CG',
                  'Cook Islands':'CK',
                  'Colombia':'CO',
                  'Comoros':'KM',
                  'Cabo Verde':'CV',
                  'Costa Rica':'CR',
                  'Cuba':'CU',
                  'Curaçao':'CW', #non-ascii?
                  'Christmas Island':'CX',
                  'Cayman Islands':'KY',
                  'Cyprus':'CY',
                  'Czechia':'CZ',
                  'Germany':'DE',
                  'Djibouti':'DJ',
                  'Dominica':'DM',
                  'Denmark':'DK',
                  'Algeria':'DZ',
                  'Dominican Republic':'DO',
                  'Ecuador':'EC',
                  'Egypt':'EG',
                  'Eritrea':'ER',
                  'Western Sahara':'EH',
                  'Spain':'ES',
                  'Estonia':'EE',
                  'Ethiopia':'ET',
                  'Finland':'FI',
                  'Fiji':'FJ',
                  'Falkland Islands (Malvinas)':'FK',
                  'France':'FR',
                  'Faroe Islands':'FO',
                  'Micronesia, Federated States of':'FM',
                  'Gabon':'GA',
                  'United Kingdom':'GB',
                  'Georgia':'GE',
                  'Guernsey':'GG',
                  'Ghana':'GH',
                  'Gibraltar':'GI',
                  'Guinea':'GN',
                  'Guadeloupe':'GP',
                  'Gambia':'GM',
                  'Guinea-Bissau':'GW',
                  'Equatorial Guinea':'GQ',
                  'Greece':'GR',
                  'Grenada':'GD',
                  'Greenland':'GL',
                  'Guatemala':'GT',
                  'French Guiana':'GF',
                  'Guam':'GU',
                  'Guyana':'GY',
                  'Hong Kong':'HK',
                  'Heard Island and McDonald Islands':'HM',
                  'Honduras':'HN',
                  'Croatia':'HR',
                  'Haiti':'HT',
                  'Hungary':'HU',
                  'Indonesia':'ID',
                  'Isle of Man':'IM',
                  'India':'IN',
                  'British Indian Ocean Territory':'IO',
                  'Ireland':'IE',
                  'Iran':'IR',
                  'Iraq':'IQ',
                  'Iceland':'IS',
                  'Israel':'IL',
                  'Italy':'IT',
                  'Jamaica':'JM',
                  'Jersey':'JE',
                  'Jordan':'JO',
                  'Japan':'JP',
                  'Kazakhstan':'KZ',
                  'Kenya':'KE',
                  'Kyrgyzstan':'KG',
                  'Cambodia':'KH',
                  'Kiribati':'KI',
                  'Saint Kitts and Nevis':'KN',
                  'Korea, South':'KR',
                  'Korea, Republic of':'KR',
                  'Kuwait':'KW',
                  'Lao People\'s Democratic Republic':'LA',
                  'Lebanon':'LB',
                  'Liberia':'LR',
                  'Libya':'LY',
                  'Saint Lucia':'LC',
                  'Liechtenstein':'LI',
                  'Sri Lanka':'LK',
                  'Lesotho':'LS',
                  'Lithuania':'LT',
                  'Luxembourg':'LU',
                  'Latvia':'LV',
                  'Macao':'MO',
                  'Saint Martin (French part)':'MF',
                  'Morocco':'MA',
                  'Monaco':'MC',
                  'Moldova':'MD',
                  'Madagascar':'MG',
                  'Maldives':'MV',
                  'Mexico':'MX',
                  'Marshall Islands':'MH',
                  'North Macedonia':'MK',
                  'Mali':'ML',
                  'Malta':'MT',
                  'Myanmar':'MM',
                  'Montenegro':'ME',
                  'Mongolia':'MN',
                  'Northern Mariana Islands':'MP',
                  'Mozambique':'MZ',
                  'Mauritania':'MR',
                  'Montserrat':'MS',
                  'Martinique':'MQ',
                  'Mauritius':'MU',
                  'Malawi':'MW',
                  'Malaysia':'MY',
                  'Mayotte':'YT',
                  'Namibia':'NA',
                  'New Caledonia':'NC',
                  'Niger':'NE',
                  'Norfolk Island':'NF',
                  'Nigeria':'NG',
                  'Nicaragua':'NI',
                  'Niue':'NU',
                  'Netherlands':'NL',
                  'Norway':'NO',
                  'Nepal':'NP',
                  'Nauru':'NR',
                  'New Zealand':'NZ',
                  'Oman':'OM',
                  'Pakistan':'PK',
                  'Panama':'PA',
                  'Pitcairn':'PN',
                  'Peru':'PE',
                  'Philippines':'PH',
                  'Palau':'PW',
                  'Papua New Guinea':'PG',
                  'Poland':'PL',
                  'Puerto Rico':'PR',
                  'Korea, Democratic People\'s Republic of':'KP',
                  'Portugal':'PT',
                  'Paraguay':'PY',
                  'Palestine, State of':'PS',
                  'French Polynesia':'PF',
                  'Qatar':'QA',
                  'Réunion':'RE',
                  'Romania':'RO',
                  'Russia':'RU',
                  'Rwanda':'RW',
                  'Saudi Arabia':'SA',
                  'Sudan':'SD',
                  'Senegal':'SN',
                  'Singapore':'SG',
                  'South Georgia and the South Sandwich Islands':'GS',
                  'Saint Helena, Ascension and Tristan da Cunha':'SH',
                  'Svalbard and Jan Mayen':'SJ',
                  'Solomon Islands':'SB',
                  'Sierra Leone':'SL',
                  'El Salvador':'SV',
                  'San Marino':'SM',
                  'Somalia':'SO',
                  'Saint Pierre and Miquelon':'PM',
                  'Serbia':'RS',
                  'South Sudan':'SS',
                  'Sao Tome and Principe':'ST',
                  'Suriname':'SR',
                  'Slovakia':'SK',
                  'Slovenia':'SI',
                  'Sweden':'SE',
                  'Eswatini':'SZ',
                  'Sint Maarten (Dutch part)':'SX',
                  'Seychelles':'SC',
                  'Syrian Arab Republic':'SY',
                  'Turks and Caicos Islands':'TC',
                  'Chad':'TD',
                  'Togo':'TG',
                  'Thailand':'TH',
                  'Tajikistan':'TJ',
                  'Tokelau':'TK',
                  'Turkmenistan':'TM',
                  'Timor-Leste':'TL',
                  'Tonga':'TO',
                  'Trinidad and Tobago':'TT',
                  'Tunisia':'TN',
                  'Turkey':'TR',
                  'Tuvalu':'TV',
                  'Taiwan*':'TW',
                  'Tanzania, United Republic of':'TZ',
                  'Uganda':'UG',
                  'Ukraine':'UA',
                  'United States Minor Outlying Islands':'UM',
                  'Uruguay':'UY',
                  'United States':'US',
                  'Uzbekistan':'UZ',
                  'Holy See (Vatican City State)':'VA',
                  'Saint Vincent and the Grenadines':'VC',
                  'Venezuela':'VE',
                  'Virgin Islands, British':'VG',
                  'Virgin Islands, U.S.':'VI',
                  'Vietnam':'VN',
                  'Vanuatu':'VU',
                  'Wallis and Futuna':'WF',
                  'Samoa':'WS',
                  'Yemen':'YE',
                  'South Africa':'ZA',
                  'Zambia':'ZM',
                  'Zimbabwe':'ZW',
                  'West Bank and Gaza':'WG'}
    return countries


def load_data():
    data_dir = "../../data/external/"
    covid_confirmed = pd.read_csv(data_dir+'time_series_covid19_confirmed_global.csv')
    covid_deaths = pd.read_csv(data_dir+'time_series_covid19_deaths_global.csv')
    covid_recovered = pd.read_csv(data_dir+'time_series_covid19_recovered_global.csv')
    #print(fetch_hopkins.fetch_hopkins_from_db())
    return covid_confirmed, covid_deaths, covid_recovered


def logistic(t, a, b, c, d):
    return c + (d - c)/(1 + a * np.exp(- b * t))


def exponential(t, a, b, c):
    return a * np.exp(b * t) + c


def plotCasesandPredict(confirmed_cases_df, country, days, current_date):
    # filter down to region rows from the country of interest
    country_cases = confirmed_cases_df[confirmed_cases_df['Country/Region'] == country]
    # take only the columns of interest (cases by date)
    date_columns = country_cases.drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis = 1)
    # sum the different regions in the country to create the time series frame
    co = date_columns.T.sum(axis = 1)
    co = pd.DataFrame(co)
    co.columns = ['Cases']
    # drop the early dates for a country, before they had a case
    co = co.loc[co['Cases'] > 0]

    y = np.array(co['Cases'])
    x = np.arange(y.size)

    # find case doubling time
    recentdbltime = float('NaN')
    if len(y) >= 7:
        current = y[-1]
        lastweek = y[-8]

        if current > lastweek:
            # print('\n** Based on Most Recent Week of Data **\n')
            # print('\tConfirmed cases on',co.index[-1],'\t',current)
            # print('\tConfirmed cases on',co.index[-8],'\t',lastweek)
            ratio = current/lastweek
            # print('\tRatio:',round(ratio,2))
            # print('\tWeekly increase:',round( 100 * (ratio - 1), 1),'%')
            dailypercentchange = round( 100 * (pow(ratio, 1/7) - 1), 1)
            # print('\tDaily increase:', dailypercentchange, '% per day')
            recentdbltime = round( 7 * np.log(2) / np.log(ratio), 1)
            # print('\tDoubling Time (represents recent growth):',recentdbltime,'days')

    ## TO-DO: Move figures to their own file
    # plt.figure(figsize=(10,5))
    # plt.plot(x, y, 'ko', label="Observed infections")

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
            ## TO-DO: Move figures to their own file
            # plt.plot(x, logistic(x, *lpopt), 'g--', label="Expected infections (logistic)")
            # print('\n** Based on Logistic Fit**\n')
            # print('\tR^2:', logisticr2)
            # print('\tDoubling Time (during middle of growth): ', round(ldoubletime,2), '(±', round(ldoubletimeerror,2),') days')
            logisticworked = True

            # make predictions
            day_now = y.size-1 # what day we are on now
            future_day = day_now + days # how many days in the future we are predicting

            preds_log = [logistic(t,lpopt[0],lpopt[1],lpopt[2],lpopt[3]) for t in list(range(day_now,future_day,1))] # do pred
            x_log = list(range(day_now,future_day,1))
            y_log = preds_log

            ## TO-DO: Move figures to their own file
            # plt.ticklabel_format(style = 'plain')
            # plt.plot(x_log, y_log, 'y--', label="Predicted infections (logistic)") # plot predictions

            # print('\n** Predicting day', future_day,'(',days,'days time)**\n')
            # print('\tPredicted number of infections (logistic growth):',round(preds_log[-1]))
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
            ## TO-DO: Move figures to their own file
            # plt.plot(x, exponential(x, *epopt), 'r--', label="Expected infections (exponential)")
            # print('\n** Based on Exponential Fit **\n')
            # print('\tR^2:', expr2)
            # print('\tDoubling Time (represents overall growth): ', round(edoubletime,2), '(±', round(edoubletimeerror,2),') days')
            exponentialworked = True

            # make predictions
            day_now = y.size-1 # what day we are on now
            future_day = day_now + days # how many days in the future we are predicting

            preds_exp = [exponential(t,epopt[0],epopt[1],epopt[2]) for t in list(range(day_now,future_day,1))] # do pred
            x_exp = list(range(day_now,future_day,1))
            y_exp = preds_exp

            ## TO-DO: Move figures to their own file
            # plt.ticklabel_format(style = 'plain')
            # plt.plot(x_exp, y_exp, 'b--', label="Predicted infections (exponential)") # plot predictions

            # print('\n** Predicting day', future_day,'(',days,'days time)**\n')
            # print('\tPredicted number of infections (exponential growth):',round(preds_exp[-1]))
    except:
        pass

    ## TO-DO: Move figures to their own file
    # plt.title(country + ' Cumulative Confirmed COVID-19 Cases. (Updated on '+current_date+')', fontsize="x-large")
    # plt.xlabel('Days', fontsize="x-large")
    # plt.ylabel('Total Confirmed Cases', fontsize="x-large")
    # plt.legend(fontsize="x-large")
    # #plt.show()
    # plt.yscale('linear')
    # plt.savefig('../../data/figures/'+country+'_linear.png')
    # #plt.yscale('log')
    # #plt.savefig('../../data/figures/'+country+'_log.png')

    if logisticworked and exponentialworked:
        if round(logisticr2,2) > round(expr2,2):
            return [ldoubletime, ldoubletimeerror, recentdbltime, lpopt, round(preds_log[-1])]
        else:
            return [edoubletime, edoubletimeerror, recentdbltime, epopt, round(preds_exp[-1])]

    if logisticworked:
        return [ldoubletime, ldoubletimeerror, recentdbltime, lpopt, round(preds_log[-1])]

    if exponentialworked:
        return [edoubletime, edoubletimeerror, recentdbltime, epopt, round(preds_exp[-1])]
    else:
        return [float('NaN'), float('NaN'), recentdbltime, float('NaN'), float('NaN')]

def main(days):
    # print(f'Start modeling COVID-19 cases based on latest data!')
    # load data
    covid_confirmed_df, covid_deaths, covid_recovered = load_data()
    country_code_dict = get_country_code()
    
    # get most_recent_date
    dates = covid_confirmed_df.keys()
    current_date = dates[-1]
    # get the most recent cases_confirmed for each country
    cases = covid_confirmed_df.iloc[:,[1,-1]].groupby('Country/Region').sum().sort_values(by = current_date, ascending = False)
    # find highly affected countries
    topcountries = cases[cases[current_date] >= 100].index.tolist()
     
    timestamp = datetime.now().isoformat()

    total_results = {"results":[],"timestamp":timestamp}
    
    # import total medical capacity
    resources_capacity_dict = {}
    resources_capacity_df = pd.read_csv('../../data/external/country_medical_capacities.csv')
    for index, row in resources_capacity_df.iterrows():
        resources_capacity_dict[row['country_code']] = row['total_capacity']

    # collect country demands
    country_list = []
    demands_list = []

    # create full output dictionary
    for country in topcountries:
        # run prediction model
        dbltime, dbltimeerr, recentdbltime, params, pred = plotCasesandPredict(covid_confirmed_df, country, days, current_date)
        # initialise dict of results
        country_results_dict = {}
        
        ## Generate dict values
        # get country code
        if country == 'US':
            country_code = country
        else:
            country_code = country_code_dict[country]
        # get number of hospital beds
        if country_code in resources_capacity_dict:
            resources_capacity = resources_capacity_dict[country_code]
        else:
            resources_capacity = 0
        # get current corona cases
        confirmed = cases.loc[country][0]
        # get current corona deaths
        if(not covid_deaths.empty):
            deaths = int(covid_deaths[covid_deaths['Country/Region']==country].iloc[:,-1].sum())
        else:
            deaths = 0
        # get current corona recovered
        if(not covid_recovered.empty):
            recovered = int(covid_recovered[covid_recovered['Country/Region']==country].iloc[:,-1].sum())
        else:
            recovered = 0
        # get 3 week predicted cases
        if not np.isnan(pred):
            confirmed_prediction_3w = int(pred) # we take the prediction
        else:
            confirmed_prediction_3w = -1 # why -1, not 0?
        # get 3 week predicted deaths and recovered
        if confirmed != 0:
            deaths_prediction_3w = int(deaths/confirmed * confirmed_prediction_3w) # use current perc deaths
            recovered_prediction_3w = int(recovered/confirmed * confirmed_prediction_3w) # use current perc recovered
        else:
            deaths_prediction_3w = -1 # why -1, not 0?
            recovered_prediction_3w = -1 # why -1, not 0?

        # populate dict
        country_results_dict['country_code'] = country_code
        country_results_dict['country_name'] = country
        country_results_dict['resources_capacity'] = int(resources_capacity)
        country_results_dict['covid19_capacity'] = int(round(resources_capacity * 0.05 * 0.5)) # <- 5% of hospital beds are ICD and HDU, 50% are filled with normal patients
        country_results_dict['confirmed'] = int(confirmed)
        country_results_dict['deaths'] = int(deaths)
        country_results_dict['recovered'] = int(recovered)
        country_results_dict['confirmed_prediction_3w'] = int(confirmed_prediction_3w)
        country_results_dict['deaths_prediction_3w'] = int(deaths_prediction_3w)
        country_results_dict['recovered_prediction_3w'] = int(recovered_prediction_3w)
        country_results_dict['resource_requirements_current'] = int(round((confirmed - deaths - recovered) * 0.15))
        country_results_dict['resource_requirements_3w'] = int(round((confirmed_prediction_3w - deaths_prediction_3w - recovered_prediction_3w) * 0.15))

        # append to master dict
        total_results['results'].append(country_results_dict)

        # append requirements/excess to demand list for the demands file -> distribution script
        demand = country_results_dict['covid19_capacity'] - country_results_dict['resource_requirements_3w']
        demands_list.append(demand)
        country_list.append(country_code)

    demands_frame = pd.DataFrame({'country':country_list, 'demand':demands_list})
    demands_frame.to_csv("../distribution/demands.csv", index=False)

    return total_results


if __name__ == "__main__":
    days = 21 # 3 weeks prediction
    results = main(days)
    # print(results)
    populate_results.populate_with_predicted_cases(results)



