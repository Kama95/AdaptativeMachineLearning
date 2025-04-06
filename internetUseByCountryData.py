import kagglehub

# Download latest version of the dataset
dataset_name = "meleknur/global-internet-usage-by-country-2000-2023"
path = kagglehub.dataset_download(dataset_name)

print(f"Path to dataset files: {path}")
