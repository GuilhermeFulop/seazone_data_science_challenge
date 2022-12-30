import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class PricePrepare:

    def __init__(self, data_name):
        self.data_name = data_name

    def get_data_price(self):

        data = pd.read_csv(self.data_name)
        data.rename(columns = {'airbnb_listing_id' : 'ad_id'}, inplace = True)

        return data

