import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class VivaPrepare:

    def __init__(self, data_name):
        self.data_name = data_name

    def get_data_viva(self):

        data = pd.read_csv(self.data_name)

        return data
