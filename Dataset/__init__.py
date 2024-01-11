import os
import pandas as pd

KAGGLE_DATASET = "jkraak/bitcoin-price-dataset"
KAGGLE_CSV_FILE = "bitcoin_2017_to_2023.csv"
KAGGLE_DOWNLOAD_DATASETS_COMMAND =  "kaggle datasets download"

PATH_TO_DATASET = "Dataset/"
DATASET_NAME = "bitcoin-price.csv"

class BitcoinData():

    def __init__(self):
        self.data = []

        self.download_dataset()
        self.read_dataset()

        self.length = len(self.data)

    def download_dataset(self):
        path_to_zip = os.path.join(PATH_TO_DATASET, DATASET_NAME)
        if os.path.exists(path_to_zip):
            print("Dataset already downloaded!")
            return

        download_command = " ".join([
            KAGGLE_DOWNLOAD_DATASETS_COMMAND, 
            "-d", KAGGLE_DATASET, 
            "-p", PATH_TO_DATASET,
            "--unzip"
        ])

        print("Downloading dataset...")
        os.system(download_command)
        print("Download complete!")

        rename_command = " ".join([
            "mv", 
            os.path.join(PATH_TO_DATASET, KAGGLE_CSV_FILE), 
            os.path.join(PATH_TO_DATASET, DATASET_NAME)
        ])

        os.system(rename_command)

    def read_dataset(self):
        path_to_dataset = os.path.join(PATH_TO_DATASET, DATASET_NAME)

        print("Reading dataset...")
        self.data = pd.read_csv(path_to_dataset)
        print("Dataset read!")

    def get_price_at(self, time):
        time = self.length - time - 1
        open_price = self.data.iloc[time]["open"]
        high_price = self.data.iloc[time]["high"]
        low_price = self.data.iloc[time]["low"]
        close_price = self.data.iloc[time]["close"]

        return (open_price + 2 * high_price + 2 * low_price + 3 * close_price) / 8
    
    def get_open_at(self, time):
        time = self.length - time - 1
        return self.data.iloc[time]["open"]
    
    def get_high_at(self, time):
        time = self.length - time - 1
        return self.data.iloc[time]["high"]
    
    def get_low_at(self, time):
        time = self.length - time - 1
        return self.data.iloc[time]["low"]
    
    def get_close_at(self, time):
        time = self.length - time - 1
        return self.data.iloc[time]["close"]
    
    def get_volume_at(self, time):
        time = self.length - time - 1
        return self.data.iloc[time]["volume"]
    
    def get_quote_asset_volume_at(self, time):
        time = self.length - time - 1
        return self.data.iloc[time]["quote_asset_volume"]
    