import os
import csv
import numpy as np  
import pandas as pd

KAGGLE_DATASET = "prasoonkottarathil/btcinusd"
KAGGLE_DOWNLOAD_DATASETS_COMMAND =  "kaggle datasets download"

PATH_TO_DATASET = "Dataset/"
DATASET_START_YEAR = 2017
DATASET_END_YEAR = 2018

def get_dataset_min_name(year):
    return f"BTC-{year}min.csv"

class BitcoinData():

    TIMESTAMP_INDEX = 0
    OPEN_INDEX = 1
    HIGH_INDEX = 2
    LOW_INDEX = 3
    CLOSE_INDEX = 4
    VOLUME_BTC_INDEX = 5
    VOLUME_USD_INDEX = 6

    def __init__(self):
        self.data = []

        self.download_dataset()
        self.read_dataset()

        self.length = len(self.data)
        print("Dataset length:", self.length)

    def download_dataset(self):
        files_exist = True
        for year in range(DATASET_START_YEAR, DATASET_END_YEAR + 1):
            path_to_csv = os.path.join(PATH_TO_DATASET, get_dataset_min_name(year))
            if not os.path.exists(path_to_csv):
                files_exist = False

        if files_exist:
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

    def read_dataset(self):
        print("Reading dataset...")

        for year in range(DATASET_END_YEAR, DATASET_START_YEAR - 1, -1):
            path_to_csv = os.path.join(PATH_TO_DATASET, get_dataset_min_name(year))
            with open(path_to_csv, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                next(reader, None)
                for row in reader:
                    row_data = []
                    row_data.append(int(row[0]))
                    row_data.append(float(row[3]))
                    row_data.append(float(row[4]))
                    row_data.append(float(row[5]))
                    row_data.append(float(row[6]))
                    row_data.append(float(row[7]))
                    row_data.append(float(row[8]))

                    self.data.append(row_data)        
        self.data = np.array(self.data)

        print("Dataset read!")

    def get_price_at(self, time):
        time = self.length - time - 1
        open_price = self.data[time][self.OPEN_INDEX]
        high_price = self.data[time][self.HIGH_INDEX]
        low_price = self.data[time][self.LOW_INDEX]
        close_price = self.data[time][self.CLOSE_INDEX]

        return (open_price + 2 * high_price + 2 * low_price + 3 * close_price) / 8
    
    def get_open_at(self, time):
        time = self.length - time - 1
        return self.data[time][self.OPEN_INDEX]
    
    def get_high_at(self, time):
        time = self.length - time - 1
        return self.data[time][self.HIGH_INDEX]
    
    def get_low_at(self, time):
        time = self.length - time - 1
        return self.data[time][self.LOW_INDEX]
    
    def get_close_at(self, time):
        time = self.length - time - 1
        return self.data[time][self.CLOSE_INDEX]
    
    def get_volume_btc_at(self, time):
        time = self.length - time - 1
        return self.data[time][self.VOLUME_BTC_INDEX]
    
    def get_volume_usd_at(self, time):
        time = self.length - time - 1
        return self.data[time][self.VOLUME_USD_INDEX]