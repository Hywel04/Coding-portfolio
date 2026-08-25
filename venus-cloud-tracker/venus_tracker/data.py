#imports
import os
import requests
import xarray as xr
from bs4 import BeautifulSoup
from urllib.parse import urljoin

#Handles fetching, downlaoding and loading of netcdf files from the jaxa website

def fetch_netcdf_links(url):
    """
    Scrapes the provided JAXA DARTS URL for NetCDF links matching filtering criteria.
    """
    filtered_netcdf_links = []
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        print(f"Webpage content from '{url}' successfully fetched and parsed.")

        netcdf_links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.endswith('.nc'):
                full_url = urljoin(url, href)
                netcdf_links.append(full_url)

        if netcdf_links:
            for link in netcdf_links:
                if '365' in link and '283' not in link:
                    filtered_netcdf_links.append(link)

            if filtered_netcdf_links:
                print("\nFound filtered NetCDF file links (containing '365' but not '283'):")
                for nc_link in filtered_netcdf_links:
                    print(nc_link)
            else:
                print("\nNo NetCDF file links containing '365' but not '283' were found.")
        else:
            print("\nNo NetCDF file links found on this page.")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the webpage: {e}")
        print("Please ensure the URL is correct and accessible.")

    return filtered_netcdf_links


def download_netcdf_files(filtered_netcdf_links, output_directory='./data/raw'):
    """
    Downloads list of NetCDF links into the target raw data directory.
    """
    os.makedirs(output_directory, exist_ok=True)

    if filtered_netcdf_links:
        print(f"Attempting to download {len(filtered_netcdf_links)} filtered NetCDF files...")
        for i, file_url in enumerate(filtered_netcdf_links):
            filename = os.path.basename(file_url)
            local_filepath = os.path.join(output_directory, filename)

            try:
                response = requests.get(file_url, stream=True)
                response.raise_for_status()

                with open(local_filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"[{i+1}/{len(filtered_netcdf_links)}] Successfully downloaded '{filename}' to '{output_directory}'.")

            except requests.exceptions.RequestException as e:
                print(f"[{i+1}/{len(filtered_netcdf_links)}] Error downloading '{filename}': {e}")
            except Exception as e:
                print(f"[{i+1}/{len(filtered_netcdf_links)}] An unexpected error occurred during download of '{filename}': {e}")
    else:
        print("No filtered NetCDF file links were found.")


def load_radiance_datasets(data_directory='./data/raw'):
    """
    Finds and loads all NetCDF files from the specified directory into xarray datasets.
    """
    if not os.path.exists(data_directory):
        print(f"Directory '{data_directory}' does not exist.")
        return []

    downloaded_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.nc')]
    downloaded_files.sort()

    radiance_datasets = []
    for filepath in downloaded_files:
        try:
            data = xr.open_dataset(filepath, decode_timedelta=True)
            radiance_datasets.append(data)
            print(f"Successfully loaded {os.path.basename(filepath)}")
        except Exception as e:
            print(f"Error loading {os.path.basename(filepath)}: {e}")

    if not radiance_datasets:
        print("No NetCDF datasets were loaded for plotting.")
    else:
        print(f"Successfully loaded {len(radiance_datasets)} NetCDF datasets.")

    return radiance_datasets