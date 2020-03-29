'''
This script fetches data from Hopkins DB at https://covid19api.com/#details and populate our MongoDB at
https://account.mongodb.com/account/login
'''
from absl import app
from absl import flags
import http.client
import mimetypes

# ------------------ Parameters ------------------- #
FLAGS = flags.FLAGS

# flags.DEFINE_string("par1", "val1", "blabla")
# flags.DEFINE_float("par2", "val2", "blabla")

def main(argv):

    #----------------- Fetching data using REST call -----------------#
    conn = http.client.HTTPSConnection("api.covid19api.com")
    payload = ''
    headers = {}
    conn.request("GET", "/all", payload, headers)
    print('Getting data from Hopkins DB...')
    res = conn.getresponse()
    data = res.read()
    print('Data was retrieved...')
    print(data.decode("utf-8"))

    # ----------------- Saving data in MongoDB -----------------#


if __name__ == "__main__":
    app.run(main)