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
         "number_cases": "40000"
      },
      {
         "country_code": "Spain",
         "number_cases": "30000"
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
