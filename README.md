# Project: Lobster Buoy Detection

This project detects lobster buoys in imagery files using two models. The output includes shapefiles, labeled maps, and text files with coordinates of the detected buoys.

## Requirements
- Python 3.12.6 (preferred)

## Steps to Run the Project

### 1. Create and Activate a Virtual Environment

First, create a Python virtual environment and activate it.

```bash
# Create a virtual environment\python3 -m venv .venv

# Activate the virtual environment (MacOS/Linux)
source .venv/bin/activate

# Activate the virtual environment (Windows)
.venv\Scripts\activate
```

### 2. Install Required Packages

Install the required Python packages using `pip`:

```bash
pip install ultralytics opencv-python geopandas shapely rasterio
```

### 3. Run the Script

Run the `NW.py` script to process imagery files:

```bash
python NW.py
```

### What the Script Does
- Uploads all imagery files, including the area of interest.
- Processes the imagery using two introduced models.
- Generates outputs in two separate folders:
  - Shapefiles of detected lobster buoys.
  - Labeled map as a JPG image.
  - Text file containing the coordinates of the lobster buoys.

---

## Input Data
Ensure that the imagery files, including the area of interest, are placed in the appropriate folder (`Imageries`) before running the script.

## Outputs
After running the script, the following outputs will be generated:
1. **Shapefiles**: Containing the detected lobster buoys.
2. **Labeled Map**: A JPG file showing the detected buoys.
3. **Coordinates**: A text file listing the coordinates of the detected lobster buoys.

---

## Contact
If you have any questions or issues, please reach out to Roozbeh@Northlightai.com.
