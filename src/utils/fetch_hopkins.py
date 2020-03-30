import http.client
import mimetypes
from pymongo import MongoClient
import urllib.parse
import json
from bson import ObjectId

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)


def fetch_hopkins_from_db():
    json_list = []
    username = urllib.parse.quote_plus("covics-19")
    password = urllib.parse.quote_plus("Coron@V!rus2020")
    client = MongoClient("mongodb+srv://" + username + ":" + password + "@cluster0-pjnfk.mongodb.net/test?retryWrites=true&w=majority")
    db = client['covics-19']
    hopkins = db["hopkins"].find()
    for elem in hopkins:
        json_list.append(elem)
    j_file = JSONEncoder().encode(json_list)
    return j_file
