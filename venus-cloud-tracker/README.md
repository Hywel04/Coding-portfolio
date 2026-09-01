# Venus Atmospheric Cloud Tracking Pipeline

A modular Python library developed for a 4th-year Master's research project investigating atmospheric dynamics and potential microbial habitability conditions in Venus's upper cloud deck. The package automates scraping, spatial preprocessing, feature tracking, and physical velocity calculation using JAXA Akatsuki Ultraviolet Imager (UVI) Level-3b NetCDF datasets.

## Key Features

* **Automated Data Scraping:** Programmatically fetches and filters JAXA Akatsuki UVI Level-3b `.nc` files via custom web scraping routines.
* **Illumination-Aware Bounding:** Dynamic auto-bounding algorithms (`get_auto_bounds`) isolate valid dayside illuminated planetary regions.
* **Feature Extraction & Preprocessing:** Uses `scikit-image` region properties and spatial filtering to extract morphometric metrics (contrast, centroid, area) from cloud features.
* **Physical Velocity Tracking:** Implements cross-correlation and relaxation-based matching algorithms to compute zonal ($u$) and meridional ($v$) cloud drift velocities in $m/s$.
* **Batch Processing & Logging:** Full CLI execution engine that generates structured CSV logs (`data/processed/`) and velocimetry vector maps.

## Tech Stack

* **Domain Processing:** `xarray`, `netCDF4`
* **Computer Vision & Metrics:** `scikit-image`, `scipy`
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`
* **Web Ingestion:** `requests`, `beautifulsoup4`

## Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/Hywel04/Coding-portfolio.git
cd venus-cloud-tracker
```

A fresh clone starts without local data; run the batch pipeline once to download the input files from the JAXA database (https://data.darts.isas.jaxa.jp/pub/pds3/extras/vco_uvi_l3_v1.1/) before opening the demo notebook.

### Set Up a Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate  # On Windows PowerShell use: .\venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Running the Full Batch Pipeline (CLI)

Execute the primary entry point to fetch data, process all available timesteps, and export vector maps and tracking metrics. The URL in `main.py` can be changed to another valid link to the JAXA dataset:

```bash
python main.py
```

Output logs and visual maps will be exported automatically into `data/processed/`.

### Demo Notebook

To explore the step-by-step algorithms using an automated verified frame pair scanner, open the demo notebook after running the batch pipeline or otherwise placing NetCDF files in `data/raw/`:

```bash
pip install jupyter
jupyter notebook notebooks/demo.ipynb
```

## Project Structure

```text
venus-cloud-tracker/
├── data/
│   ├── raw/                # Downloaded NetCDF datasets (.nc)
│   └── processed/          # Exported tracking logs (.csv) and vector maps (.png)
├── notebooks/
│   └── demo.ipynb          # Interactive pipeline walkthrough & visualization
├── venus_tracker/          # Core Python Package
│   ├── __init__.py
│   ├── data.py             # Scraping & NetCDF ingestion engines
│   ├── processing.py       # Spatial bounds, filtering, & feature extraction
│   ├── tracker.py          # Physical velocity tracking logic
│   ├── pipeline.py         # Frame pair execution pipeline
│   ├── visualization.py    # Vector map rendering
│   └── batch.py            # Batch orchestration manager
├── .gitignore
├── main.py                 # CLI entry point for full pipeline execution
├── README.md
└── requirements.txt
```

## Project Origin & Academic Context

This repository represents the codebase developed as part of a 4th-year Master’s research project. The project adapted and engineered cloud-tracking computer vision pipelines to analyze atmospheric dynamics on Venus—specifically evaluating localized cloud-deck transport mechanisms relevant to investigating the potential habitability and transport of microorganisms in the Venusian cloud layer (~50–65 km altitude). The pipeline processes Level-3b data captured by the Akatsuki spacecraft's Ultraviolet Imager (UVI) at $365\text{ nm}$. By tracking moving features in Venus's upper cloud deck (~65 km altitude) across sequential image frames ($\Delta t$), the software isolates the strong retrograde zonal super-rotation speeds and meridional transport dynamics, while allowing for analysis of the temporal evolution of cloud features present in the upper atmosphere.

## Academic Attribution & Acknowledgements

* **Institution:** Department of Physics & Astronomy, Cardiff University
* **Degree Program:** MPhys Integrated Masters in Astrophysics
* **Supervision:** Supervised by Professor Jane Greavs
* **Data Provider:** Datasets provided by the JAXA Akatsuki UVI team via the DARTS archive.
