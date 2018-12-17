
# coding: utf-8

# # Contains code for multi-year flood cat model simulations.
# 
# This code takes what's in section 2 of cat_model_demonstration.ipynb and loops over it many times. Each loop corresponds to a week, and the number of loops is specified by the number of weeks in the simulation.  
# 
# Steps involved: 
# 1. Load data: For event generation, local intensity calculation. 
# 2. Specify collection of properties we want to use--only properties in Phoenicia, or properties across the entire region? 
# 3. Set up structure to hold the output.  
# 4. Loop through all the weeks in the output structure. For each week, use catastrophe model to calculate damage from the weekly event. 
# 5. Save the output structure to a .csv file for future analysis. 

# In[1]:


# Import modules, set key paths. 
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from shapely.geometry import Polygon,Point,LineString,MultiPolygon
from scipy import interpolate
import time
import datetime
plt.style.use('dark_background')
#plt.style.use('seaborn-dark')

data_path = '/Users/Lucien/Documents/Learn_ML/projects/cat_flood_toy_model/selected_data'


# # Step 1. Load the data. 
# 

# In[2]:


# Functions for timing. Useful but isn't necessary for the simulations. 
def tic(): 
    return time.time()

def toc(t0):
    #print('time =',time.time()-t0,'seconds')
    return time.time()-t0

# Class with functions for saving/loading data from .pkl files. 
from toy_model_module import save_load_pkl

# Load data: 

# 1. Event generator. 
weekly_max_gh = pd.read_csv(os.path.join(data_path,'weekly_max_gage_heights.csv'),usecols=[1,2,3])

# 2. DEM, shapefiles for streets, creeks, and addresspoints. 
local_dem = save_load_pkl.load_obj(os.path.join(data_path,'local_DEM.pkl'))
zoom_x,zoom_y,zoom_elev,zoom_bounds,default_crs = local_dem['dem_X'],local_dem['dem_Y'],local_dem['dem_Z'],local_dem['zoom_bounds'],local_dem['crs']
zoom_polygon = Polygon([(zoom_bounds['min_x'],zoom_bounds['min_y']), 
                       (zoom_bounds['min_x'],zoom_bounds['max_y']),
                       (zoom_bounds['max_x'],zoom_bounds['max_y']),
                       (zoom_bounds['max_x'],zoom_bounds['min_y'])])
creeks = gpd.read_file(os.path.join(data_path,'creeks.shp'))
streets = gpd.read_file(os.path.join(data_path,'streets.shp'))
adpts = gpd.read_file(os.path.join(data_path,'address_points.shp'))

# 3. CSV file that includes x, y, z, name, and creek point ID for each point in the creek shapes. 
creek_points = pd.read_csv(os.path.join(data_path,'creek_points.csv'))
creek_points = creek_points.drop('Unnamed: 0', axis=1)

# 4. CSV file used to map each point on the DEM to the nearest point on the creek. 
# Contains x, y, z_DEM, z for nearest creek point, ID for nearest creek point, name of nearest creek. 
dem2creek = pd.read_csv(os.path.join(data_path,'dem2creek.csv'))
dem2creek = dem2creek.drop('Unnamed: 0', axis=1)


# # Step 2. Focus on only Phoenicia? 

# In[3]:


# Set only_phoenicia equal to true if we only want to focus on Phoenicia. 
# Also shave off columns we don't need for the simulation. 
only_phoenicia = True  
if only_phoenicia:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        foc_adpts = adpts[(adpts['Unincorpor']=='Phoenicia')]
        foc_adpts = foc_adpts[['NYSAddress','PropertyTy','CostOfRepl','geometry']]
else: 
    foc_adpts = adpts[['NYSAddress','PropertyTy','CostOfRepl','geometry']]
    


# # Step 3. Set up output structure. 
# Description of output structure: a dataframe with five columns. Will store the start date, end date, max gage height on the esopus, max gage height on the stony clove, and the total damage during each week of the simulation. 

# In[4]:


# Number of years to simulate
sim_yrs = 3000

# Simulation start and end times. 
time_st = datetime.datetime(year=1,month=1,day=1,hour=0,minute=0,second=0)
time_ed = datetime.datetime(year=sim_yrs,month=12,day=31,hour=23,minute=59,second=59)

# Time interval for each loop--one week. 
time_interval = datetime.timedelta(weeks=1)

# Used to separate the end time of one week from the start time of the following week. 
dt = datetime.timedelta(seconds=1)

# Number of steps/weeks in the simulation
num_steps = (time_ed-time_st)//time_interval

# Setup output dataframe. 
week_st,week_ed,gh_e,gh_s,total_damage = [],[],[],[],[]
for i in range(num_steps): 
    week_st.append(time_st+(i*time_interval))
    week_ed.append(time_st+((i+1)*time_interval)-dt)
    gh_e.append(np.nan)
    gh_s.append(np.nan)
    total_damage.append(np.nan)
    
output = pd.DataFrame({
    'step_st': week_st,
    'step_ed': week_ed,
    'gh_e': gh_e,
    'gh_s': gh_s,
    'tot_damage': total_damage
})    


# # Step 4. Simulate using a loop. 
# Loop through all the weeks specified by the output file. For each week: 
# * Generate weekly event (gage heights on stony brook and esopus at time of maximum $gh_e^2+gh_s^2$)
# * Calculate local intensity (inundation at each property during flood). 
# * Calculate damage to each property from the inundation and property's cost of replacement. 
# * Calculate total damage to all properties. 
# * Place gh_e, gh_s, total damage in the output structure. 

# In[5]:


# Function for event generation
from toy_model_module import generate_week

# Function for calculating sign of number (+/-)
sign = lambda x: (1, -1)[x < 0]

# Function for determining water level of creek after merge betwen esopus and stony clove. 
def merged_water_level(ghe,ghs,sign):
    if sign(ghe*ghs) == 1:
        ghm = sign(ghe)*np.sqrt(ghe**2+ghs**2)
    elif sign(ghe*ghs) == -1:
        sum_sqr = sign(ghe)*(ghe**2)+sign(ghs)*(ghs**2)
        ghm = sign(sum_sqr)*np.sqrt(np.abs(sum_sqr))
    return ghm

# Import other classes/functions
from toy_model_module import flooding_fcns # Class for calculating flood extent, inundation
flood_fcns = flooding_fcns(zoom_x,zoom_y)
from toy_model_module import damage_fcns # Class of damage functions. 

# To speed up the simulations, I save the inundation at each address point for each weekly 
# flooding event to a dictionary. Then, when the event is repeated (this will happen because 
# the event generator only draws from a set of ~520 weekly events), I use these inundations 
# instead of recalculating them. This reduces time for each loop by about 80%. 
inund_dict = {'gh_e': [], 'gh_s': [], 'dam_adpts': []}

# Loop through all weeks in the simulation
for i,week in output.iterrows():
        
    # ==============================================================================
    # Generate event for week. 
    month_of_week = (week['step_st']+(0.5*time_interval)).month
    ghe,ghs = generate_week(weekly_max_gh,month_of_week=month_of_week)
    flood = {'esopus': ghe, 'stony_clove': ghs}
    flood['merged'] = merged_water_level(ghe,ghs,sign)
    
    # ==============================================================================
    # Calculate inundation at each property. 
    
    # Determine if ghe,ghs combination is already in the dictionary of events. 
    dict_ind = [i for i, x, y in zip(list(range(len(inund_dict['gh_e']))),inund_dict['gh_e'],inund_dict['gh_s']) if x == ghe and y == ghs]
    
    # If event is not already in dictionary, do calculations and add to dictionary. 
    # If so, just use inundations from the dictionary. 
    if not dict_ind: 
        
        # Calculate water level at each point on the creeks. 
        creek_points2 = flood_fcns.determine_z_flood(creeks,creek_points,flood)

        # Assign a water level to each DEM point--water level at stream point that's nearest 
        # the DEM point. 
        df_map = creek_points2[['crkpt_id','Z_flood']]
        dem2creek2 = pd.merge(dem2creek, df_map, left_on='crkpt_id', right_on='crkpt_id')

        # Using the water levels, determine whether each point on DEM grid floods. If so, 
        # also determine the inundation at that point. 
        dem2creek2['gd_inund'] = dem2creek2['Z_flood']-dem2creek2['Z']
        dem2creek2['gd_inund'] = dem2creek2['gd_inund'].apply(lambda x: max(x,0))
        dem2creek2['flooded'] = dem2creek2['gd_inund']>0

        # Create 2D arrays of flooding status, inundation, which are on DEM grid. 
        zoom_flooded, zoom_inundat = flood_fcns.flood_to_dem_grid(dem2creek2[['X','Y','flooded','gd_inund']])

        # Interpolate with 2D array of inundation to calculate the inundation at each address point. 
        dam_adpts = flood_fcns.inundation_at_adpts(zoom_inundat,foc_adpts)

        # Append to inundation data to dicationary. 
        inund_dict['gh_e'].append(ghe)
        inund_dict['gh_s'].append(ghs)
        inund_dict['dam_adpts'].append(dam_adpts[dam_adpts['gd_inund']>0])
                
    else: 
                
        dam_adpts = inund_dict['dam_adpts'][dict_ind[0]]
        

    # ==============================================================================
    # Calculate the damage function, damage value (USD) at each affected property. 
    # add to address points dataframe. 

    damage = []
    for _,row in dam_adpts.iterrows():
        dfcn = damage_fcns(row['gd_inund'],row['CostOfRepl'],row['PropertyTy'])
        dam = dfcn.damage_fcn2()
        damage.append(dam)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dam_adpts['damage_fcn_value'] = pd.Series(damage, index=dam_adpts.index)
        dam_adpts['damage_USD'] = dam_adpts['damage_fcn_value']*dam_adpts['CostOfRepl']
    
    
    # ==============================================================================
    # Sum to determine the total damage across all properties. 
    tot_damage = sum(dam_adpts['damage_USD'])
    
    
    # ==============================================================================
    # Enter ghe, ghs, and total damage (USD) into output structure. 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output['gh_e'][i],output['gh_s'][i],output['tot_damage'][i] = ghe,ghs,tot_damage
        

# Save the output structure to a CSV file. File name reflects the time at which the simulation 
# was run. Format: output_YYYY_mm_dd_HH_MM.csv. Creates a directory to store the output file 
# one does not already exist. 
if not os.path.isdir('./output'): 
    os.mkdir('./output')
current_time = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M')
output.to_csv(os.path.join('./output','output_'+current_time+'.csv'))

