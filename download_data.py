import kagglehub
import pandas as pd
import shutil
import os

print("Downloading dataset...")
path = kagglehub.dataset_download("suraj520/customer-support-ticket-dataset")

print("Path to dataset files:", path)

# The dataset is likely a CSV file, let's copy it to the current directory
for f in os.listdir(path):
    shutil.copy(os.path.join(path, f), os.path.join(".", f))
    
print("Successfully copied dataset to local directory.")
