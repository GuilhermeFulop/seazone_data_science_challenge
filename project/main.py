import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from DetailsPrepare import DetailsPrepare
from PricePrepare import PricePrepare
from VivaPrepare import VivaPrepare
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import math
import time
from Functions import *

start = time.time()

text = 'Getting data'
print('=' * 3 *len(text))
print('='*len(text)+text+'='*len(text))
print('=' * 3 *len(text) + '\n')

details = DetailsPrepare('Details_Data.csv')
details  = details.get_details_data()
viva =VivaPrepare('VivaReal_Itapema.csv')
viva = viva.get_data_viva()
price = PricePrepare('Price_AV_Itapema-001.csv')
price = price.get_data_price()
mesh = pd.read_csv('Mesh_Ids_Data_Itapema.csv')

mesh.rename(columns = {'airbnb_listing_id' : 'ad_id'}, inplace = True)

price['date'] = pd.to_datetime(price['date'])
price.rename(columns = {'airbnb_listing_id' : 'ad_id'}, inplace = True)
viva['address_neighborhood'] = viva['address_neighborhood'].str.lower()

text = 'Getting neighborhoods'
print('=' * 3 *len(text))
print('='*len(text)+text+'='*len(text))
print('=' * 3 *len(text) + '\n')

neighborhood = get_location(mesh)

text = 'Getting date and merging prices'
print('=' * 2 *len(text))
print('='*len(text)+text+'='*len(text))
print('=' * 2 *len(text) + '\n')

max_date, min_date, details = date_prepare(details)
details = details.merge(neighborhood, on = 'ad_id', how = 'left')
price = get_price_date(price, max_date, min_date)
price_backup = price[['ad_id', 'date', 'price']]
details = details.merge(price_backup, on = ['ad_id', 'date'])
details_backup = create_details_dackup(details)

text = 'Getting the 10% most wanted houses'
print('=' * 3 *len(text))
print('='*len(text)+text+'='*len(text))
print('=' * 3 *len(text) + '\n')

most_wanted = most_wanted_properties(details, price)
most_wanted_ad_id = most_wanted.index

df = duplicate_eraser_price_selector(details_backup, most_wanted_ad_id)
most_wanted_suburb, percent_sub, percent_sub_with_char, most_wanted_bedrooms, most_wanted_bathrooms, percent_char = most_wanted_characteristics(df)

print(f"""
Answer to question one: the best property profile to invest profile to invest is with {most_wanted_bedrooms} bedrooms and {most_wanted_bathrooms} bathrooms.
Considering the properties with more than 90% of occupation rate between {min_date} and {max_date} there are {percent_char}% houses with these characteristics.
The neighborhood with the highest amount of houses is {most_wanted_suburb.title()}, 
there are {percent_sub}%, and this neighborhood has the highest amount of properties with the best profile,
about {percent_sub_with_char}%.
""")

house_distribution(df, 'suburb')
house_distribution(df, 'price')
house_distribution(df, 'number_of_bedrooms')
house_distribution(df, 'number_of_bathrooms')

text = 'Working with Viva dataset'
print('=' * 3 *len(text))
print('='*len(text)+text+'='*len(text))
print('=' * 3 *len(text) + '\n')

text = 'Adjusting neighborhoods names'
print('=' * 3 *len(text))
print('='*len(text)+text+'='*len(text))
print('=' * 3 *len(text) + '\n')

viva = neighborhood_adjustment(viva)

text = 'Visualizing outliers'
print('=' * 3 *len(text))
print('='*len(text)+text+'='*len(text))
print('=' * 3 *len(text) + '\n')

boxplot_plotter(viva, 'yearly_iptu', 'no')
boxplot_plotter(viva, 'monthly_condo_fee', 'no')

viva = outlier_detector(viva, 'yearly_iptu')
viva = outlier_detector(viva, 'monthly_condo_fee')

text = 'Visualizing new data'
print('=' * 3 *len(text))
print('='*len(text)+text+'='*len(text))
print('=' * 3 *len(text) + '\n')

boxplot_plotter(viva, 'yearly_iptu', 'yes')
boxplot_plotter(viva, 'monthly_condo_fee', 'yes')

text = 'Finding the best location in terms of revenue'
print('=' * 3 *len(text))
print('='*len(text)+text+'='*len(text))
print('=' * 3 *len(text) + '\n')

viva = adjusting_iptu_and_monthly_fee(viva)

airbnb_houses, viva_houses = get_number_of_houses(mesh, neighborhood, viva)

viva_maintain = maintain_houses(viva, 'address_neighborhood')

airbnb_year_income = year_income(details)

viva_year_cost = year_outcome(viva_maintain)

revenue = highest_revenue(airbnb_year_income, airbnb_houses, viva_year_cost, viva_houses)

print(f""" \n
Answers to question 2 and 3: The best location in terms of revenue is {revenue['suburb'][0].title()}.
This happens because the sale price for each house tends to be {round(revenue['house_sales_price'][0]/revenue['house_sales_price'][2], 2) * 100}% lower than for 'Meia Praia', even though the anual income is similar.
'Morretes' has a cost and sale price similar, but the income is {round(revenue['income/house'][0]/revenue['income/house'][1], 2) * 100}% higher, this makes the revenue be {round(revenue['revenue'][0]/revenue['revenue'][1], 2)} times greater than the revenue for Morretes.
So, the reasons that make Casa Branca have the highest revenue value is because it has the highest income per house and the lowest sales price per house and also it has the second lowest cost values
""")

text = 'Fiding building location'
print('=' * 3 *len(text))
print('='*len(text)+text+'='*len(text))
print('=' * 3 *len(text) + '\n')

neighborhood_cost = dict(zip(revenue['suburb'], revenue['cost/house']))

data_final = cheaper_square_meter(viva)

data_final = cheaper_total_cost(data_final, neighborhood_cost)

parking_spaces, parking_spaces_percent = get_parking_spaces(viva)

print(f""" \n
Answer to question 4: There are some allotment land to sell in Viva Real dataset. The best place to build it is the lot with 
listing_id = {data_final.loc[0, 'listing_id']} because it has the lowest square meter value, even if we consider the the total cost. 

The apartments should be designed as was found in question one: apartments with {most_wanted_bedrooms} bedrooms and {most_wanted_bathrooms} bathrooms. By the informations in the dataset, the ideal parking space is for {parking_spaces} cars, {parking_spaces_percent}% of all the dataset has this size for garage.

Although 'Casa Branca' is the best neighborhood in terms of revenue, the lot avaiable are way more expensive than for this land. 
""")

text = 'Fiding occupation rate and mean price'
print('=' * 3 *len(text))
print('='*len(text)+text+'='*len(text))
print('=' * 3 *len(text) + '\n')

final_occupation_rate = total_occupation(details, price, neighborhood, 'morretes')

mean_price = get_price(details)

year_interval = [2024, 2025, 2026]
dic = {}
for i in year_interval:
    year = i
    data = income(mean_price, final_occupation_rate, i, data_final)
    dic[year] = data

    df = pd.DataFrame(list(dic.items()),
                       columns=['year', 'name'])
    df[['return', 'profit']] = pd.DataFrame(df['name'].tolist(), index=df.index)
    df.drop('name', axis = 1, inplace = True)

print('\n')
print(df)

print(f"""\n
The dataframe above shows the return and the proft for 2024, 2025 and 2026, we can see that in 2024, the return is R${df['return'][0]}, the proft is R${df['profit'][0]}. This happens because the total value invested in buying the lot and constructing the building is not totally payed.

But in 2025, the return is R${df['return'][1]} and the profit is R${df['profit'][1]}, that shows that the building is is generating a positive profit. 

In 2026, the return is R${df['return'][2]} and the profit is R${df['profit'][2]}.
""")

final = time.time()

print(f"Total time = {int((final - start)//60)} min.")