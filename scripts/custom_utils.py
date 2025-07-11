import matplotlib.pyplot as plt
import math

import geopandas as gpd
from shapely.geometry import Point


# Constants
global EARTH_RADIUS
EARTH_RADIUS = 6.371e6

class utils:

    def __init__(self):
        pass

    @staticmethod
    def deg_to_rad(deg):
        '''
        Convert degrees to radians
        '''

        return (deg*math.pi)/180
    
    @staticmethod
    def rad_to_deg(rad):
        '''
        Convert radians to degrees
        '''

        return (rad/math.pi)*180

    @staticmethod
    def plot(thresholder_obj, year):
        '''
        Plot thresholding results for all clusters in a given year
        '''

        feature_arr, control_arr, feature_err_arr, control_err_arr, feature_n_arr = [], [], [], [], []
        for cluster in range(4):
            thresholder_obj.set_data_params(year, cluster)
            feature_res, feature_error_res, feature_n, control_res, control_error_res = thresholder_obj.calculate()
            feature_arr.append(feature_res)
            control_arr.append(control_res)
            feature_err_arr.append(feature_error_res)
            control_err_arr.append(control_error_res)
            feature_n_arr.append(feature_n)

        thresholds = thresholder_obj.thresholds

        fig, axs = plt.subplots(2, 2, figsize = (12, 10))
        
        plt.subplots_adjust(hspace = 0.3, wspace = 0.3)
        feature_name = ' '.join(thresholder_obj.feature.split('_')).title()
        fig.suptitle(f'New York City {feature_name}: {year}', fontsize = 15)
        
        r, c = 0, 0 
        for cluster in range(4):

            axs[r][c].errorbar(thresholds*EARTH_RADIUS, feature_arr[cluster], yerr = feature_err_arr[cluster], fmt='o', c = 'blue', label = 'Treatment')
            axs[r][c].errorbar(thresholds*EARTH_RADIUS, control_arr[cluster], yerr = control_err_arr[cluster], fmt='o', c = 'red', label = 'Control')

            axs[r][c].set_title(f'Cluster {cluster} (n = {feature_n_arr[cluster]})')
            axs[r][c].set_xlabel('Radius (m)')
            axs[r][c].set_ylabel('Daily Crime Density (count / km^2)')
            axs[r][c].set_ylim(bottom = 0)

            if c == 0:
                c += 1
            else:
                c = 0
                r += 1

        handles, labels = axs[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, ncols = 2, loc = 'upper center', bbox_to_anchor = (0.5, 0.95))

        return fig
    
    @staticmethod
    def label_tract(point, shapefile, radians = True):
        '''
        Return tract label of single point, given as (lat, lon) coordinate pair
            - Important that shapefile columns are labeled correctly: 'tract_id' and 'geometry'
            - Make sure to specify if point is given in degrees or radians - it must be converted to degrees before calling sjoin

        Originally used for crime rate calculations, no longer called 
        '''

        if radians:
            point = Point(utils.rad_to_deg(point[1]), utils.rad_to_deg(point[0]))
        else:
            point = Point(point[1], point[0])
        
        gdf = gpd.GeoDataFrame(geometry = [point])
        gdf.crs = 'EPSG:4326'
        gdf = gdf.to_crs('EPSG:4326')
        labeled_gdf = gdf.sjoin(shapefile[['tract_id', 'geometry']], how = 'left', predicate = 'intersects')

        return labeled_gdf['tract_id'].loc[0]