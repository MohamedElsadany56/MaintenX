import pandas as pd
import requests
from io import StringIO

class DataLoader:
    def __init__(self):
        self.data_urls = {
            'telemetry': 'https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_telemetry.csv',
            'errors': 'https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_errors.csv',
            'maintenance': 'https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_maint.csv',
            'failures': 'https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_failures.csv',
            'machines': 'https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_machines.csv'
        }
    
    def load_data(self, data_type):
        """Load data from URL or local cache"""
        url = self.data_urls[data_type]
        response = requests.get(url)
        return pd.read_csv(StringIO(response.text))
    
    def load_all_data(self):
        """Load all datasets"""
        datasets = {}
        for key in self.data_urls.keys():
            datasets[key] = self.load_data(key)
        return datasets