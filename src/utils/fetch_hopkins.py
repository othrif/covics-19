from pymongo import MongoClient
import urllib.parse
import json
from bson import ObjectId
import pandas as pd


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)


def _fetch_hopkins_from_db():
    '''
    This script fetches latest Hopkins data from our MongoDB and eturns them as list of dictionaries
    :return: List of dictionaries containing Hopkins data
    '''
    entries_list = []
    username = urllib.parse.quote_plus("covics-19")
    password = urllib.parse.quote_plus("Coron@V!rus2020")
    client = MongoClient("mongodb+srv://" + username + ":" + password + "@cluster0-pjnfk.mongodb.net/test?retryWrites=true&w=majority")
    db = client['covics-19']
    hopkins = db["hopkins"].find()
    for elem in hopkins:
        entries_list.append(elem)
    # entries_json = JSONEncoder().encode(entries_list) # use this to return in str format
    return entries_list

def load_model_data():
    '''
    This script fetches latest Hopkins data from our MongoDB to feed our prediction model
    :return:
    '''
    entries_list = _fetch_hopkins_from_db()
    df = pd.DataFrame(entries_list)         # DataFrame of all Hopkins cases

    df_confirmed = df.loc[df.Status == 'confirmed']
    df_deaths = df.loc[df.Status == 'deaths']
    df_recovered = df.loc[df.Status == 'recovered']

    # we don't groupby lat and lon ---> hopkins mismatches on lat and lon values are therefore avoided
    return df_confirmed.reset_index().groupby(['Province', 'Country', 'Date'])['Cases'].aggregate('first').unstack(), \
           df_deaths.reset_index().groupby(['Province', 'Country', 'Date'])['Cases'].aggregate('first').unstack(), \
           df_recovered.reset_index().groupby(['Province', 'Country', 'Date'])['Cases'].aggregate('first').unstack()
