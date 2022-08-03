from .base import AbstractDataset

import pandas as pd

from datetime import date


class beautyDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'beauty'

    @classmethod
    def url(cls):
        return 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['reviews_Beauty_5.json.gz']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings_Beauty.csv')
        df = pd.read_csv(file_path, sep=',', header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df


