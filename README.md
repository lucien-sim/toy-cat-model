# README for Project 1: Catastrophe Model for Flooding in Phoenicia, NY

To learn more about catastrophe modeling and geospatial analysis in Python, I created a toy catastrophe flood model. This repository contains everything that's necessary to understand and run the toy model. 

## Getting started
1. Download the repository, place in a folder/directory. 
2. Make sure you have python 3 installed and have the software required to open jupyter notebooks. Also make sure you have the dependencies installed (see below). I recommend using anaconda/jupyter lab for all this.
3. Familiarize yourself with everything that's in the repository (see below). Note that the documentation for the entire model is in 'cat_model_demonstration.ipynb', including a detailed description and demonstration of how the model works.
4. Then you should be ready to run code in all the notebooks and scripts in repository. 

## What are the dependencies? 
os, pandas, numpy, seaborn, datetime, geopandas, matplotlib, rasterio, shapely, scipy, pickle, warnings, time. 

## Roadmap: what's in all the files and directories?
**Directories:** 
* **selected_data:** contains all the data that's necessary to run the catastrophe model. This includes a 10-meter Digital Elevation Model (DEM), a shapefile with stream/river locations, a shapefile with stream locations, a shapefile with addresspoint locations, a shapefile road/street locations, a csv file with information on all the points used to define the creeks in their shapefile, and a csv file linking each point on the DEM to the nearest point on a creek. 
* **output:** Contains output csv file for a 3000-year simulation with the toy model. Output from future multiyear simulations will also be saved to this directory. 

**Files:**
* **cat_model_demonstration.ipynb:** This is a detailed write-up of the catastrophe model.
    * Provides a brief background on the project and on catastrophe modeling, in general. 
    * Explains, in detail, how monetary loss is calculated for a single flood event. 
    * Produces several important risk metrics from output from the 3000-year simulation, which would be useful for assessing Phoenicia's flood risk. 
    * Explains how the toy model could be improved, if revised to provide accurate assessments of risk. 
* **multiyear_simulation.py:** Python script used to run multiyear simulations. 
* **multiyear_simulation.ipynb:** Notebook outlining the multiyear simulation script in a readable format. Code in this jupyter notebook is identical to the code in multiyear_simulation.py. 
* **toy_model_module.py:** Module that contains functions, classes that are used in the catastrophe model. 
* **extract_selected_data.ipynb:** Notebook that walks you through the process of extracting data required from the catastrophe model from the files you can download online. **IMPORTANT NOTE:** This code isn't necessary for running the catastrophe model; I've just included it in case you want to extract the data on your own. Although I don't know why you would--I've done it for you. 

## How to I simulate a single flood event? 
cat_model_demonstration.ipynb walks you through this. 

## How do I run a multiyear simulation? 
Change the 'sim_years' variable value to the number of desired simulation years in 'multiyear_simulation.py'. Then run the script form the terminal. The model requires approximately 15 seconds per simulated year. 

If you have any questions, feel free to contact me at lucien.simpfendoerfer@gmail.com. 
