#Imports
import os
import matplotlib
matplotlib.use('Agg')  # Headless batch run - no display needed for saved PNGs

from venus_tracker.data import fetch_netcdf_links, download_netcdf_files, load_radiance_datasets
from venus_tracker.batch import VenusBatchTracker

#Database can be found at https://data.darts.isas.jaxa.jp/pub/pds3/extras/vco_uvi_l3_v1.1/
#Naviagate to the data folder and select the netcdf folder to get the data for the pipeline, it should be a list of netcdf files that can be downloaded and processed by the pipeline.
#See the example DATASET_URL below for the correct path to the netcdf files.

# Configuration Constants
DATASET_URL = 'https://data.darts.isas.jaxa.jp/pub/pds3/extras/vco_uvi_l3_v1.1/vcouvi_7012/data/l3b/netcdf/r0275/'
RAW_DATA_DIR = './data/raw'

def main():
    print("--- Starting Venus Cloud Tracker Pipeline ---")
    
    # 1. Scraping Data Links from Source URL
    links = fetch_netcdf_links(url=DATASET_URL)
    
    # 2. Downloading NetCDF Datasets to Local Directory
    download_netcdf_files(filtered_netcdf_links=links, output_directory=RAW_DATA_DIR)
    
    # 3. Loading NetCDF Datasets into Memory
    datasets = load_radiance_datasets(data_directory=RAW_DATA_DIR)
    
    # 4. Processing Pipeline
    if datasets:
        print(f"\nPipeline ready. Initializing batch processing for {len(datasets)} datasets...")
        output_path = os.path.join("data", "processed", "venus_tracking_log.csv")
        batch_runner = VenusBatchTracker(datasets, output_csv=output_path)
        batch_runner.run_all()
        for dataset in datasets:
            dataset.close()
    else:
        print("\nPipeline stopped: No valid datasets loaded.")

if __name__ == "__main__":
    main()

