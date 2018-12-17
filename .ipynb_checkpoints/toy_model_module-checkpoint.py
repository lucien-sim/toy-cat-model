# Module for all the functions/classes used in the toy catastrophe model. 
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
import pandas as pd
import geopandas as gpd
from rasterio import features
from rasterio.transform import from_origin
from shapely.geometry import MultiPolygon, Polygon
from scipy import interpolate


class save_load_pkl: 
    """
    Class for saving and loading .pkl files. 
    """
        
    def __init__(self):
        self
    
    def save_obj(obj,name): 
        """
        Function for saving .pkl files. 
        obj = object that you want to save. 
        name = filepath/filename to which you want to save the object. 
        """
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
    def load_obj(name):
        """
        Function for loading data from .pkl files
        name = filepath/filename from which you want to load the object. 
        """
        with open(name, 'rb') as f:
            return pickle.load(f)
        

def plot_phoenicia_base(ax,zoom_x,zoom_y,zoom_elev,topocmap,vminmax,creeks,streets,zoom_bounds,only_phoenicia=False): 
    """
    Function that plots the basis for every flooding event around Phoenicia
    """
    
    # Hillshade function. 
    def hillshade(array, azimuth, angle_altitude):
        # Source: http://geoexamples.blogspot.com.br/2014/03/shaded-relief-images-using-gdal-python.html
        x, y = np.gradient(array)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            slope = np.pi/2. - np.arctan(np.sqrt(x**2 + y**2))
        aspect = np.arctan2(-x, y)
        azimuthrad = azimuth*np.pi / 180.
        altituderad = angle_altitude*np.pi / 180.
        shaded = np.sin(altituderad) * np.sin(slope)          + np.cos(altituderad) * np.cos(slope)          * np.cos(azimuthrad - aspect)
        return 255*(shaded + 1)/2
    
    extent = xmin,xmax,ymin,ymax = min(zoom_x[0,:]),max(zoom_x[0,:]),min(zoom_y[:,0]),max(zoom_y[:,0])
    ax.matshow(hillshade(zoom_elev, 30, 30), extent=extent, cmap='Greys', alpha=.3,zorder=10)
    CS = ax.contourf(zoom_x,zoom_y,zoom_elev,levels=30,cmap=topocmap,vmin=vminmax[0], vmax=vminmax[1])
    creeks.plot(ax=ax,color='#1D64F3')
    streets.plot(ax=ax,color='#696969')
    ax.xaxis.tick_bottom()
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    if only_phoenicia: 
        ax.set_xlim((555500,558500))
        ax.set_ylim((4657500,4660000))
    else: 
        ax.set_xlim((zoom_bounds['min_x'],zoom_bounds['max_x']))
        ax.set_ylim((zoom_bounds['min_y'],zoom_bounds['max_y']))
    return CS

def generate_week(weekly_max_gh,month_of_week=1):
    in_month = (weekly_max_gh['month']==month_of_week)
    inds_in_month = in_month[in_month].index
    ind_choose = np.random.choice(inds_in_month)
    ghe = weekly_max_gh['esopus'][ind_choose]
    ghs = weekly_max_gh['stony_clove'][ind_choose]
    return ghe, ghs


class flooding_fcns: 
    
    """
    Class that holds functions for: 
    1. Determining water level at each creek point. 
    2. Converting flood status in list of DEM points back to DEM grid. 
    3. Creating multipolygon to show flooding on map.  
    4. Calculation inundation at a list of properties. 
    """
    
    def __init__(self,zoom_x,zoom_y): 
        """
        zoom_x, zoom_y: meshgrid for the DEM data. 
        """
        self.zoom_x = zoom_x
        self.zoom_y = zoom_y
        

    def determine_z_flood(self,creeks,creek_points,flood):
        """ 
        Function for calculating water level at each creek point. 
        Input: 
        creeks = geodataframe containg shapefiles for the esopus, stony clove, 
                 and esopus-after-merge creeks. 
        creek_points = dataframe containing a list of all creek points. Includes columns for: 
                       x coordinate, y coordinate, elevation, creek name, and creek point ID
                       for each point on all creeks. 
        flood = dictionary with information on the flooding in each creek. 
        Output: creek_points dataframe with added column that contains elevation of new water 
                level for each point on the creeks. 
        """

        # Height of flood at intersection point. 
        x_inter,y_inter = creeks['geometry'][1].coords[-1][0],creeks['geometry'][1].coords[-1][1]
        crk_pt_inter = creek_points[(creek_points['X'] == x_inter)].iloc[2,:]
        z_lake = crk_pt_inter['Z']+flood['merged']

        # Calculate the water level at each creek point. 
        z_flood = np.zeros((np.shape(creek_points)[0],1))
        for index,row in creek_points.iterrows():
            if row['crk_nm'] == 'esopus': 
                z_flood[index] = max(row['Z']+flood['esopus'],z_lake)
            elif row['crk_nm'] == 'stony_clove': 
                z_flood[index] = max(row['Z']+flood['stony_clove'],z_lake)
            elif row['crk_nm'] == 'merged': 
                fade_dist = 750 # m
                fact = min(np.sqrt((row['X']-x_inter)**2+(row['Y']-y_inter)**2),fade_dist)/fade_dist
                z_flood[index] = fact*(row['Z']+flood['merged'])+(1-fact)*z_lake

        # Add column with water levels to creek_points dataframe, return dataframe. 
        creek_points['Z_flood'] = z_flood

        return creek_points


    def flood_to_dem_grid(self,data_to_grid):
        """
        Maps data stored with a list of DEM points back to the DEM grid. 
        Input: 
        data_to_grid: pandas dataframe with x coordinate, y coordinate, and the data values of interest 
                      for each point. 
        Output: 
        zoom_flooded: Grid showing whether flooding occurred at each point specified by zoom_x and zoom_y. 
        zoom_inundat: Grid showing the ground inundation (in m) at each point specified by zoom_x and zoom_y. 
        """

        # The mapping--merge left based on X/Y coordinates of each point. 
        dem_pt_df = pd.DataFrame({'X': self.zoom_x.ravel(), 'Y': self.zoom_y.ravel()})
        plot_flood_df = pd.merge(dem_pt_df, data_to_grid, how='left' ,left_on=['X','Y'], right_on = ['X','Y'])

        # Fill in values for points more than 15 m above the nearest creek point. These were not included 
        # step 2 above to save computational time. 
        plot_flood_df['flooded'] = plot_flood_df['flooded']>0
        plot_flood_df['gd_inund'] = plot_flood_df['gd_inund'].apply(lambda x: max(0,x))

        # Convert to array. 
        zoom_flooded = np.asarray(plot_flood_df['flooded']).reshape(np.shape(self.zoom_x))
        zoom_inundat = np.asarray(plot_flood_df['gd_inund']).reshape(np.shape(self.zoom_x))

        return zoom_flooded,zoom_inundat


    def create_flood_multipolygon(self,zoom_flooded):
        """
        Create multipolygons for the regions that flood. 
        Input: 
        zoom_flooded = grid showing where on DEM grid flooding occurred. 
        Output: 
        flood_pgon_gdf = geodataframe with multipolygon geometry for the flooding. 
        """

        # Create affine transformation matrix.         
        x = self.zoom_x[0,:].ravel()
        y = self.zoom_y[:,0].ravel() 
        res = (x[1] - x[0])
        transform = from_origin(x[0],y[0], res, res)

        # Create multipolygon, store in geodataframe. 
        flood_pgons = []
        for geom, val in features.shapes(zoom_flooded.astype('int32'),transform=transform):
            if val > 0: 
                pgon_coords = Polygon(geom['coordinates'][0])
                flood_pgons.append(pgon_coords)
        flood_pgons = MultiPolygon(flood_pgons)
        flood_pgon_gdf = gpd.GeoDataFrame(pd.DataFrame({'geometry': flood_pgons}))

        return flood_pgon_gdf


    def inundation_at_adpts(self,zoom_inundat,adpts):
        """
        This function determines whether flooding occurs and calculates the inundation at a series of addresspoints. 
        Input: 
        zoom_inundat = ground inundation on DEM grid. 
        adpts = dataframe containing AT LEAST x,y coordinates for each addresspoint. 
        Output: 
        adpts = same dataframe, but with added column for the flood status, inundation at each addresspoint. 
        """

        # Calculate inundation at each addresspoint using linear interpolation. 
        flooded,ground_inundation = [],[]
        inund_interp = interpolate.RegularGridInterpolator(points=(np.flipud(self.zoom_y[:,0]),self.zoom_x[0,:]),values=np.flipud(zoom_inundat),method='linear')
        for _,row in adpts.iterrows():
            pt_geom = row['geometry']
            try: 
                inund = float(inund_interp((pt_geom.coords[0][1],pt_geom.coords[0][0])))
            except: 
                inund = 0
            ground_inundation.append(inund)
            if inund>0: 
                flooded.append(True)
            else: 
                flooded.append(False)
            
        # Add inundation depths and whether property expereinced flooding to addresspoints dataframe. 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adpts['flooded'] = pd.Series(flooded, index=adpts.index)
            adpts['gd_inund'] = pd.Series(ground_inundation, index=adpts.index)

        return adpts


class damage_fcns: 
    
    def __init__(self,wd,cor,prop_type): 
        self.wd = wd
        self.cor = cor
        self.prop_type = prop_type
        
        
    def damage_fcn1(self,dop_mean=0.3,dop_std=0.15): 
        """
        Damage functions for commercial/residential properties. Calculates the damage/cost_of_replacement
        from the inundation depth (wd), degree of preparation (between 0 and 1), and the property's 
        cost_of_replacement. 

        Math for markdown cell describing damage_fcn1. Saved it here because it's annoying to recreate. 
        $$
        \begin{equation}
        DR = min\Bigg( \Big[ \frac{wd}{10} \Big] ^{\frac{1}{2}} * \big[1-0.5*dop \big] * \Big[1-0.3* \Big( \frac{cor}{500000} \Big) \Big], 1\Bigg)
        \label{damageresidential}
        \tag{1}
        \end{equation}
        $$     

        $$
        \begin{equation}
        DR = min\Bigg( \Big[ \frac{wd}{7} \Big] * \big[1-0.25*dop \big] * \Big[1-0.3* \Big( \frac{cor}{500000} \Big) \Big], 1\Bigg)
        \label{damagecommercial}
        \tag{2}
        \end{equation}
        $$

        """
        
        def degree_of_preparation(mean,std):
            """
            Mean and std are for the distribution of prep times, which are distributed normally. Keep 
            in mind that only degrees of prep between 0 and 1 are valid. 
            Each time the function is called, degrees of prep are generated randomly until one falls
            between 0 and 1. When this happens, the degree of preparation is returned. 
            """
            val = 3
            while val < 0 or val > 1: 
                np.random.RandomState(seed=None)
                val = np.random.normal(loc=mean,scale=std)
            return val
        
        # Calculate degree of preparation
        dop = degree_of_preparation(dop_mean, dop_std)
        
        # Compute damage function value. 
        if self.prop_type == 'residential':
            damage_amt = min(np.sqrt(self.wd/10)*(1-0.5*dop)*(1-0.3*(self.cor/500000)),1)
        elif self.prop_type == 'commercial':
            damage_amt = min((self.wd/7)*(1-0.25*dop)*(1-0.3*(self.cor/500000)),1)  
        return damage_amt

    
    def damage_fcn2(self,return_mean=False): 
        """
        Updated version of the damage function. 
        """
        if self.wd > 0: 
            if self.prop_type == 'residential':
                DR_mean = min(np.sqrt(self.wd/10)*(1-0.3*(self.cor/500000)),1)
            elif self.prop_type == 'commercial':
                DR_mean = min((self.wd/7)*(1-0.3*(self.cor/500000)),1) 

            if not return_mean: 
                DR = DR_mean+max(min(np.random.normal(loc=0,scale=0.15),1-DR_mean),-DR_mean)
            else: 
                DR = DR_mean
                
        else: 
            DR = 0
                
        return DR

    
def show_damfcn2(damage_fcns):
    """
    Plot to demonstrate damage function 2 in the damage_fcns class above. 
    """
    wd2test = np.linspace(0,15,47)
    wd_all,dam_all = [],[]
    for wd in wd2test: 
        for i in range(500):
            dfcn = damage_fcns(wd,250000,'residential')
            wd_all.append(wd)
            dam_all.append(dfcn.damage_fcn2())

    mean_dam = []
    for wd in wd2test: 
        dfcn = damage_fcns(wd,250000,'residential')
        mean_dam.append(dfcn.damage_fcn2(return_mean=True))

    plt.figure(figsize=(9, 5))
    CS = plt.hist2d(wd_all,dam_all,bins=[np.linspace(0,15,46),np.linspace(0,1,21)],cmap='Greys',
                    cmin=0,cmax=0.4,normed=True)
    plt.plot(wd2test,mean_dam,'r-',label='Mean DR')
    plt.xlabel('Inundation (m)')
    plt.ylabel('Damage ratio')
    plt.legend()
    plt.grid(True,color='#DCDCDC')
    plt.colorbar(CS[3])
    plt.show()
    
    
class risk_assessment:
    
    """
    Class used to calculate various useful risk metrics. 
    1. Average annual loss
    2. Standard deviation of annual losses
    3. Coefficient of variation for annual losses
    4. Exceedance probability curve
    5. Approximate return period for specified loss
    6. Appoximate loss corresponding to specified return period
    
    Required for initialization:
    yearly_loss = dataframe with columns "year" and "loss" (in million USD). 

    """
    
    def __init__(self,yearly_loss): 
        self.yearly_loss = yearly_loss
        
    def avg_annual_loss(self):
        return self.yearly_loss['loss'].mean()
    
    def std_annual_loss(self):
        return self.yearly_loss['loss'].std()
    
    def cv_annual_loss(self):
        return self.yearly_loss['loss'].std()/self.yearly_loss['loss'].mean()
    
    def ep_curve(self,num_yrs=1): 
        """
        Function for creating exceedance probability curve. 
        INPUT: 
        num_yrs = years in timeframe over which we want the exceedance probabilities. 
        OUTPUT: 
        Figure and axis handles for plot. Figure should also be plotted. 
        """
        # Calculate multi-year losses
        yr_st = self.yearly_loss['year'][0]
        yr_ed = self.yearly_loss['year'].iloc[-1]
        multiyear_losses = []
        for st in range(yr_st-1,yr_ed-num_yrs-1,num_yrs):
            multiyear_losses.append(self.yearly_loss['loss'].iloc[st:st+num_yrs].sum())
        multiyear_losses.append(self.yearly_loss['loss'].iloc[-(num_yrs+1):].sum())

        # Calculate exceedance probabilities for multiyear losses
        x_loss = np.linspace(0,np.ceil(max(multiyear_losses)),100)
        p_exceed = []
        for loss in x_loss: 
            p_exceed.append((multiyear_losses>=loss).sum()/len(multiyear_losses))

        # Plot curve. 
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        plt.plot(x_loss,p_exceed,color='green',linewidth=2.5)
        ax.set_xlabel('Loss (million USD)')
        ax.set_ylabel('Exceedance probability')
        ax.set_xlim([0,max(x_loss)])
        ax.grid(True)
        plt.show()

        return fig,ax
    
    
    def loss2returnpd(self,spec_loss):
        """
        Function to calculate the return period of a specified loss. Returns a 'inf' if the specified loss is 
        above the range of yearly losses--we don't have data to estimate the return period of that loss. 
        INPUT: 
        spec_loss = loss for which we want to calculate the return period (same unit as loss data)
        OUTPUT: 
        return_period = return period, rounded to one decimal place. 
        """

        # Calculate probabilities of exceeding several threshold losses
        x_loss = np.linspace(0,np.ceil(max(self.yearly_loss['loss'])),100)
        p_exceed = [(self.yearly_loss['loss']>=loss).sum()/len(self.yearly_loss['loss']) for loss in x_loss]

        # Interpolate to estimate exceedance probability for given loss
        if spec_loss <= max(x_loss):
            interp_exc = interpolate.interp1d(x_loss,p_exceed)
            p_exceed_sl = interp_exc(spec_loss)
            return_period = np.around(1/p_exceed_sl,decimals=1)
        else: 
            return_period = np.inf

        return return_period
    
    
    def returnpd2loss(self,return_period):
        """
        Function to calculate the loss corresponding to a particular return period. Returns 'inf' if the 
        return period is larger than the number of years for which we have data--we don't have data to 
        estimate the return period. 
        INPUT: 
        return_pd = return period for which we want to calculate the corresponding loss
        OUTPUT: 
        retn_loss = loss corresponding to specified return period. 
        """
        x_loss = np.linspace(0,np.ceil(max(self.yearly_loss['loss'])),100)
        p_exceed = [(self.yearly_loss['loss']>=loss).sum()/len(self.yearly_loss['loss']) for loss in x_loss]
        if return_period<=len(self.yearly_loss['loss']):
            interp_loss = interpolate.interp1d(p_exceed,x_loss)
            retn_pd_loss = interp_loss(1/return_period)
        else: 
            retn_pd_loss = np.inf

        return retn_pd_loss
