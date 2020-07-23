'''
This script populate DB with predicted number of cases per country.
'''

from pymongo import MongoClient
import urllib.parse
import json

'''
# example of input data
predictions_dict = {
  "results":[
    {
      "country_code": "IT",
      "country_name": "Italy",
      "resources_capacity": "39000",
      "confirmed": "40000",
      "deaths": "3000",
      "recovered": "800",
      "confirmed_prediction_3w": "50000",
      "deaths_prediction_3w": "4000",
      "recovered_prediction_3w": "900"
    },
    {
      "country_code": "ES",
      "country_name": "Spain",
      "resources_capacity": "40000",
      "confirmed": "30000",
      "deaths": "2500",
      "recovered": "600",
      "confirmed_prediction_3w": "60000",
      "deaths_prediction_3w": "2800",
      "recovered_prediction_3w": "1200"
      }
   ],
   "timestamp": "2020-03-29 17:05:51.514470"
}'''

def populate_with_predicted_cases(predictions_dict):
    '''
    This function uploads predictions per country to MongoDB
    :param predictions_dict: dictionary of predictions per country
    :return: An instance of InsertOneResult.
    '''
    username = urllib.parse.quote_plus("covics-19")
    password = urllib.parse.quote_plus("Coron@V!rus2020")
    client = MongoClient("mongodb+srv://" + username + ":" + password + "@cluster0-pjnfk.mongodb.net/test?retryWrites=true&w=majority")
    db = client['covics-19']
    predictions_col = db["predictions"]
    result = predictions_col.insert_one(predictions_dict)
    return result

def populate_with_distributions(distributions_dict):
    '''
    This function uploads predictions per country to MongoDB
    :param populate_with_distributions: dictionary of predictions per country
    :return: An instance of InsertOneResult.
    '''
    username = urllib.parse.quote_plus("covics-19")
    password = urllib.parse.quote_plus("Coron@V!rus2020")
    client = MongoClient("mongodb+srv://" + username + ":" + password + "@cluster0-pjnfk.mongodb.net/test?retryWrites=true&w=majority")
    db = client['covics-19']
    distributions_col = db["distributions"]
    result = distributions_col.insert_one(distributions_dict)
    return result
