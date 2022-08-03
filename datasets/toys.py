from .base import AbstractDataset

import pandas as pd

from datetime import date


class toysDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'toys'

    @classmethod
    def url(cls):
        return 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Toys_and_Games.csv'

    @classmethod
    def zip_file_content_is_folder(cls):
        return False

    @classmethod
    def all_raw_file_names(cls):
        return ['ratings_Toys_and_Games.csv']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings_Toys_and_Games.csv')
        df = pd.read_csv(file_path, sep=',', header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df


