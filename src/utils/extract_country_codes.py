'''
From country names to country codes
'''

import pycountry
import csv

country_codes_list = []

with open('../../data/external/beds_capacity.csv') as f:
    csv_reader = csv.reader(f, delimiter=',')
    for row in csv_reader:
        country = pycountry.countries.search_fuzzy(row[0])
        country_codes_list.append([country[0].alpha_2, row[1]])

    with open('../../data/external/beds_capacity_country_codes.csv', 'w', newline='') as out:
        writer = csv.writer(out)
        for elem in country_codes_list:
            writer.writerow(elem)