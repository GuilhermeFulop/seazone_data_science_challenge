import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deep_translator import GoogleTranslator
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm

def date_prepare(data):

    data['date'] = data['ano'].map(str)+ \
                            '-'+data['mes'].map(str)+\
                            '-'+data['dia'].map(str)

    data['date'] = pd.to_datetime(data['date'])
    max_date = max(data['date'])
    min_date = min(data['date'])

    return max_date, min_date, data

def get_location(data):

    neighborhood = data[['ad_id','latitude', 'longitude']].copy()
    neighborhood['location'] = neighborhood['latitude'].map(str)+','+neighborhood['longitude'].map(str)
    locator = Nominatim(user_agent = 'myGeocoder', timeout = 10)
    rgeocode = RateLimiter(locator.reverse, min_delay_seconds=0.001)
    neighborhood['address'] = tqdm(neighborhood['location'].apply(lambda x : rgeocode(x)))
    neighborhood['suburb'] = neighborhood['address'].apply(lambda x : x.raw['address'].get('suburb'))
    neighborhood['suburb'] = neighborhood['suburb'].str.lower()
    neighborhood.drop(['latitude', 'longitude', 'location', 'address'], axis = 1, inplace = True)

    return neighborhood

def get_price_date(data, max_date, min_date):

    data['date'] = pd.to_datetime(data['date'])
    data = data[(data['date'] <= max_date) & (data['date'] >= min_date)]
    data.drop_duplicates(subset = ['ad_id', 'date'], inplace = True)  

    return data

def most_wanted_properties(data1, data2):
    details_backup2 = data1[['ad_id', 'date', 'price', 'number_of_bathrooms', 'number_of_bedrooms', 'number_of_beds', 'suburb']].\
                    groupby(['ad_id', 'date','price','number_of_bathrooms', 'number_of_bedrooms', 'number_of_beds']).sum().reset_index()
    price2 = data2[['ad_id', 'date', 'av_for_checkin']]

    details_backup2 = pd.merge(details_backup2, price2, on = ['ad_id', 'date'])
    details_backup2['occupied'] = details_backup2['av_for_checkin'].map({True: 1, False : 0})

    occupation = details_backup2.groupby('ad_id').count()
    tx = details_backup2[['ad_id', 'occupied']].groupby('ad_id').sum()

    occupation['tx'] = tx['occupied']
    occupation['rate'] = occupation['tx']/occupation['occupied'] * 100 
    
    most_wanted = occupation[(occupation['rate'] >= 90) & (occupation['occupied'] >= 9)]

    return most_wanted


def create_details_dackup(data):

    df = data[['ad_id', 'suburb', 'date', 'price', 'number_of_bathrooms', 
                           'number_of_bedrooms', 'number_of_beds']]\
                            .groupby(['ad_id', 'suburb', 'price','number_of_bathrooms', 
                                      'number_of_bedrooms', 'number_of_beds']).sum().reset_index()
    return df

def duplicate_eraser_price_selector(data, index):

    df = data[data['ad_id'].isin(index)]
    df = df[df['price'] < 30000]
    df = df.drop_duplicates(subset = 'ad_id')

    return df

def most_wanted_characteristics(data):

    df = data[['number_of_bedrooms', 'number_of_bathrooms', 'ad_id']].groupby(['number_of_bedrooms', 'number_of_bathrooms']).count()
    df = df.sort_values(by = 'ad_id', ascending = False)
    df['percent'] = round(df['ad_id']/df['ad_id'].sum() * 100, 2)
    percent_char = df.loc[df.index[0], 'percent']
    bedrooms, bathrooms = df.index[0]

    df = data[['suburb','ad_id']].groupby(['suburb']).count()
    df = df.sort_values(by = 'ad_id', ascending = False)
    df['percent'] = round(df['ad_id']/df['ad_id'].sum() * 100, 2)
    suburb = df.index[0]
    percent_sub = df.loc[suburb, 'percent']
    
    df = data[['suburb','number_of_bedrooms', 'number_of_bathrooms', 'ad_id']].groupby(['suburb','number_of_bedrooms', 'number_of_bathrooms']).count()
    df = df.sort_values(by = 'ad_id', ascending = False)
    df['percent'] = round(df['ad_id']/df['ad_id'].sum() * 100, 2)
    houses = df['percent'][0]

    return suburb, percent_sub, houses, bedrooms, bathrooms, percent_char

def house_distribution(data, column):
    sub = data[column].value_counts().to_dict()
    
    fig = plt.figure(figsize = (25, 10))
    sns.countplot(data = data, x = str(column), palette = 'rocket', order = sub.keys())
    plt.title(f'{column.title()} of the 10% most wanted houses', fontsize = 22, pad = 10)
    plt.xlabel(f'{column.title()}', fontsize = 22)
    plt.ylabel('Amount', fontsize = 22)
    plt.tick_params(labelsize=18)
    plt.grid(True, axis = 'y')
    plt.xticks(rotation = 90);
    if column == 'price':
        plt.xlim(-0.5, 15.5)
    plt.savefig(f'{column.title()} of the 10% most wanted houses.png', format = 'png')


def neighborhood_adjustment(data):

    data.loc[data['address_neighborhood'] == 'meia praia - frente mar', 'address_neighborhood'] = 'meia praia'
    data.loc[data['address_neighborhood'] == 'zona 1', 'address_neighborhood'] = 'meia praia'
    data.loc[data['address_neighborhood'] == 'andorinha', 'address_neighborhood'] = 'meia praia'
    data.loc[data['address_neighborhood'] == 'horizon palace', 'address_neighborhood'] = 'meia praia'
    data.loc[data['address_neighborhood'] == 'praia mar', 'address_neighborhood'] = 'meia praia'

    return data

def adjusting_iptu_and_monthly_fee(data):

    x = data['address_neighborhood'].value_counts()
    x = x.to_dict()
    address_neighborhood_a_100 = []
    address_neighborhood_u_100 = []
    c = 0
    for key in x.keys():
        if x.get(key) >= 100:
            address_neighborhood_a_100.append(key)
        else:
            address_neighborhood_u_100.append(key)

    data = data.dropna(subset = ['sale_price'])
    data['yearly_iptu'].fillna(1, inplace = True)
    data['monthly_condo_fee'].fillna(1, inplace = True)

    for i in address_neighborhood_a_100:
        
        ind1 = data[(data['address_neighborhood'] == i) \
                    & (data['unit_type'] == 'APARTMENT') \
                    & (data['monthly_condo_fee'].isin([n for n in range(0, 5)]))].index
        ind2 = data[(data['address_neighborhood'] == i) \
                    & (data['unit_type'] == 'APARTMENT') \
                    & (data['yearly_iptu'].isin([n for n in range(0, 5)]))].index
        ind3 = data[(data['address_neighborhood'] == i) \
                    & (data['unit_type'] == 'HOME') \
                    & (data['monthly_condo_fee'].isin([n for n in range(0, 5)]))].index
        ind4 = data[(data['address_neighborhood'] == i) \
                    & (data['unit_type'] == 'HOME') \
                    & (data['yearly_iptu'].isin([n for n in range(0, 5)]))].index

        data.loc[data.index.isin(ind1), 'monthly_condo_fee'] = data[(data['address_neighborhood'] == i) & (data['unit_type'] == 'APARTMENT')]['monthly_condo_fee'].mean()
        data.loc[data.index.isin(ind2), 'yearly_iptu'] = data[(data['address_neighborhood'] == i) & (data['unit_type'] == 'APARTMENT')]['yearly_iptu'].mean()
        data.loc[data.index.isin(ind3), 'monthly_condo_fee'] = data[(data['address_neighborhood'] == i) & (data['unit_type'] == 'HOME')]['monthly_condo_fee'].mean()
        data.loc[data.index.isin(ind4), 'yearly_iptu'] = data[(data['address_neighborhood'] == i) & (data['unit_type'] == 'HOME')]['yearly_iptu'].mean()

        
        mean_apt_fee = data[(data['unit_type'] == 'APARTMENT')]['monthly_condo_fee'].mean()
        mean_apt_iptu = data[(data['unit_type'] == 'APARTMENT')]['yearly_iptu'].mean()
        mean_home_fee = data[(data['unit_type'] == 'HOME')]['monthly_condo_fee'].mean()
        mean_home_iptu = data[(data['unit_type'] == 'HOME')]['yearly_iptu'].mean()

    for i in address_neighborhood_u_100:
        
        ind1 = data[(data['address_neighborhood'] == i) \
                    & (data['unit_type'] == 'APARTMENT')].index
        ind2 = data[(data['address_neighborhood'] == i) \
                    & (data['unit_type'] == 'APARTMENT')].index
        ind3 = data[(data['address_neighborhood'] == i) \
                    & (data['unit_type'] == 'HOME')].index
        ind4 = data[(data['address_neighborhood'] == i) \
                    & (data['unit_type'] == 'HOME')].index
        
        data.loc[data.index.isin(ind1), 'monthly_condo_fee'] = mean_apt_fee
        data.loc[data.index.isin(ind2), 'yearly_iptu'] = mean_apt_iptu
        data.loc[data.index.isin(ind3), 'monthly_condo_fee'] = mean_home_fee
        data.loc[data.index.isin(ind4), 'yearly_iptu'] = mean_home_iptu

    return data

def boxplot_plotter(data, column, treated):

    fig = plt.figure(figsize = (25, 10))
    sns.boxplot(data = data, x = column, palette = 'rocket')
    plt.title(f'{column.title()} distribution', fontsize = 18, pad = 10)
    plt.xlabel(f'{column.title()}', fontsize = 18)
    plt.tick_params(labelsize=15)
    plt.grid(True, axis = 'y')
    plt.xticks(rotation = 45);
    if treated == 'no': 
        plt.savefig(f'{column.title()} distribution.png', format = 'png')
    elif treated == 'yes':
        plt.savefig(f'{column.title()} distribution corrected.png', format = 'png')

def outlier_detector(data, column):

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1 #Interquartile range
    fence_low  = Q1 - 1.5 * IQR
    fence_high = Q3 + 1.5 * IQR
    data_out = data.loc[(data[column] > fence_low) & (data[column] < fence_high)]

    return data_out

def maintain_houses(data, column):

    x = data['address_neighborhood'].value_counts()
    x = x.to_dict()
    address_neighborhood_a_50 = []
    address_neighborhood_u_50 = []

    for key in x.keys():
        if x.get(key) >= 50:
            address_neighborhood_a_50.append(key)
        else:
            address_neighborhood_u_50.append(key)

    df_maintain = data.loc[(data['address_neighborhood'].isin(address_neighborhood_a_50))]

    return df_maintain

def get_number_of_houses(data1, data2, data3):
    
    # getting number of houses per neighborhood in Airbnb data
    df = pd.merge(data1, data2, on = 'ad_id')
    df = df.drop_duplicates(subset = ['ad_id'])
    airbnb_houses = df.groupby('suburb').count().reset_index()
    airbnb_houses.rename(columns = {'index' : 'suburb', 'ad_id' : 'houses'}, inplace = True)
    airbnb_houses.drop(['latitude', 'longitude', 'aquisition_date', 'ano', 'mes', 'dia'], inplace = True, axis = 1)
    airbnb_houses['suburb'] = airbnb_houses['suburb'].str.lower()
    
    # getting number of houses per neighborhood in Viva Real data
    viva_houses = data3['address_neighborhood'].value_counts().reset_index()
    viva_houses.rename(columns = {'index' : 'address_neighborhood', 'address_neighborhood' : 'houses'}, inplace = True)

    return airbnb_houses, viva_houses

def year_income(data):
    
    data['suburb'] = data['suburb'].str.lower()
    airbnb_year_income = data[['suburb', 'price']].groupby('suburb').sum()
    
    return airbnb_year_income

def year_outcome(data):
    
    viva_year_cost = data[['address_neighborhood', 'sale_price', 'yearly_iptu', 'monthly_condo_fee']].groupby('address_neighborhood').sum()
    
    return viva_year_cost

def highest_revenue(airbnb_year_income, airbnb_houses, viva_year_cost, viva_houses):
    
    airbnb_year_income = pd.merge(airbnb_year_income, airbnb_houses, on = 'suburb', how = 'left')
    viva_year_cost = pd.merge(viva_year_cost, viva_houses, on = 'address_neighborhood', how = 'left')
    
    airbnb_year_income['year_income'] = airbnb_year_income['price'] * 365 * 0.8 #considerando uma taxa de ocupação de 80%
    airbnb_year_income['year_income_by_house'] = airbnb_year_income['year_income']/airbnb_year_income['houses']
    
    viva_year_cost['year_cost'] = viva_year_cost['yearly_iptu'] + viva_year_cost['monthly_condo_fee'] * 12
    viva_year_cost['year_cost_per_house'] = viva_year_cost['year_cost']/viva_year_cost['houses']
    viva_year_cost['sale_price_per_house'] = viva_year_cost['sale_price']/viva_year_cost['houses']
    
    airbnb_year_income = airbnb_year_income.reset_index()
    viva_year_cost = viva_year_cost.reset_index()
    viva_year_cost.rename(columns = {'address_neighborhood' : 'suburb'}, inplace = True)
    
    final = pd.merge(airbnb_year_income[['suburb', 'year_income_by_house']], 
                     viva_year_cost[['suburb', 'year_cost_per_house', 'sale_price_per_house']], 
                     on = 'suburb', 
                     how = 'left')
    final['revenue'] = (final['year_income_by_house'] - final['year_cost_per_house'])/final['sale_price_per_house']
    final = final.sort_values(by = 'revenue', ascending = False).reset_index(drop = True)
    
    final.rename(columns = {'year_income_by_house' : 'income/house',
                            'year_cost_per_house' : 'cost/house',
                            'sale_price_per_house' : 'house_sales_price'}, 
                            inplace = True)

    return final

def drop_condo(data):
    
    data['listing_desc'] = data['listing_desc'].str.lower()
    df = data[data['listing_desc'].str.contains('condomínio', na = False)]
    index = df.index

    return index

def cheaper_square_meter(data):

    aux = data[(data['unit_type'].isin(['ALLOTMENT_LAND', 'RESIDENTIAL_BUILDING', 'RESIDENTIAL_ALLOTMENT_LAND'])) & \
           data['address_neighborhood'].isin(['morretes', 'casa branca', 'meia praia'])]
    aux = aux[['listing_id', 
               'listing_desc', 
               'sale_price', 
               'address_neighborhood', 
               'unit_type', 
               'usable_area', 
               'total_area']]
    aux['listing_desc'] = aux['listing_desc'].str.lower()

    index = drop_condo(aux)
    aux = aux.drop(index)
    
    aux['usable_area'].fillna(0, inplace = True)
    aux['total_area'].fillna(0, inplace = True)
    aux['usable_area_c'] = aux['usable_area']
    aux['total_area_c'] = aux['total_area']
    aux.loc[(aux['usable_area_c'] == 0), 'usable_area_c'] = aux['total_area']
    aux.loc[(aux['total_area_c'] == 0), 'total_area_c'] = aux['usable_area_c']
    aux['area'] = (aux['usable_area_c'] + aux['total_area_c']) / 2
    aux['m2_value'] = round(aux['sale_price'] / aux['area'], 2)
    aux =  aux.sort_values(by = 'm2_value', ascending = True)
    aux.drop(['usable_area', 'total_area', 'usable_area_c', 'total_area_c'], axis = 1, inplace = True)

    return aux

def cheaper_total_cost(data, dict_name):
    
    for row in data.index:
    
        data.loc[row, 'cost'] = round(dict_name.get(data.loc[row, 'address_neighborhood']), 2)
    
    data['total'] = data['m2_value'] / data['cost']
    data = data.sort_values(by = 'total').reset_index(drop = True)
    
    return data

def get_parking_spaces(data):

    aux = data[['parking_spaces', 'listing_id']].groupby(['parking_spaces']).count()
    aux['percent'] = round(aux['listing_id'] * 100/aux['listing_id'].sum(), 2)
    aux = aux.sort_values(by = 'percent', ascending = False).reset_index()
    parking_spaces = aux.loc[0][0] 
    amount = aux.loc[0][2]

    return parking_spaces, amount

def total_occupation(data1, data2, data3, neighborhood_name):
    
    details_backup2 = data1[['ad_id', 'date', 'price']].\
                groupby(['ad_id', 'date','price']).sum().reset_index()
    price2 = data2[['ad_id', 'date', 'av_for_checkin']]

    details_backup2 = pd.merge(details_backup2, price2, on = ['ad_id', 'date'])
    details_backup2['occupied'] = details_backup2['av_for_checkin'].map({True: 1, False : 0})

    occupation = details_backup2.groupby('ad_id').count()
    tx = details_backup2[['ad_id', 'occupied']].groupby('ad_id').sum()

    occupation['tx'] = tx['occupied']
    occupation['rate'] = occupation['tx']/occupation['occupied']
    occupation = occupation.reset_index()
    occupation = pd.merge(occupation, data3, on = 'ad_id')
    occupation = occupation.groupby('suburb').mean().reset_index()
    occupation['suburb'] = occupation['suburb'].str.lower()
    occupation = occupation[occupation['suburb'] == neighborhood_name].reset_index(drop = True)
    
    final_rate = round(occupation['rate'][0], 2)
    
    return final_rate

def get_price(data):
    
    aux = data.groupby('suburb').mean().reset_index()
    aux = aux[aux['suburb'] == 'morretes'].reset_index(drop = True)
    price = round(aux['price'][0], 2)
    
    return price

def income(mean_price, final_occupation_rate, year, data):

    year = year - 2022
    year = [d for d in range(0, year)]

    income_airbnb_2023 = mean_price * final_occupation_rate * 365 * 50 * 0.85 * 0.6
    income_rental_2023 = 0.4 * 50 * 2500 * 12
    outcome_building = data['sale_price'][0] + 5 * 1300 * 2000
    outcome_fees = data['cost'][0] * 0.6 * 50

    total_return = 0
    for i in year:
        if i == 0:            
            house_anual_income = (income_airbnb_2023 + income_rental_2023 - outcome_fees)
            remaining_building_price = outcome_building - house_anual_income
            if remaining_building_price < 0:
                remaining_building_price = 0
            total_return += house_anual_income
            proft = total_return - remaining_building_price
        elif i != 0:
            income_airbnb_2023 = income_airbnb_2023 * 1.1
            income_rental_2023 = income_rental_2023 * 1.1
            outcome_fees = outcome_fees * 1.1        
            house_anual_income = (income_airbnb_2023 + income_rental_2023 - outcome_fees) + house_anual_income
            remaining_building_price = remaining_building_price - house_anual_income
            if remaining_building_price < 0:
                    remaining_building_price = 0
            remaining_building_price = outcome_building - house_anual_income
            total_return += house_anual_income
            proft = total_return - remaining_building_price

    return round(total_return, 2), round(proft, 2)