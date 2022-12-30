import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



class DetailsPrepare:

    def __init__(self, data_name):
        self.data_name = data_name

    def get_details_data(self):
        data = pd.read_csv(self.data_name)
        return data





