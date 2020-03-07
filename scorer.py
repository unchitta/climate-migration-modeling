import numpy as np
import pandas as pd
import pickle



def compatibility(c1,c2):
    
    def _common_crops(c1,c2):
        island_crops = pd.read_csv('data/compatibility_data/island_crops.csv')
        country_crops = pd.read_csv('data/compatibility_data/country_crops.csv')
        df = pd.merge(island_crops[island_crops['Area']==c1],\
                      country_crops[country_crops['Area']==c2],\
                      how='inner', on='Item Code')
        return df.shape[0] / 173
    
    def _latitude_diff(c1,c2):
        latitude = pd.read_csv('data/compatibility_data/latitude.csv')
        return abs(latitude.loc[c1]['Latitude'] - latitude.loc[c2]['Latitude']) / 180
    
    def _religion_sim(c1,c2):
        island_religions = {'Marshall Islands':'Christian',
                            'Tuvalu':'Christian',
                            'Maldives':'Muslim',
                            'Kiribati':'Christian'}
        country_religions = pd.read_csv('data/compatibility_data/religion_pct_by_country.csv')
        
        religion = island_religions[c1]
        
        return country_religions.loc[c2][religion] / 100
    
    return np.average([_common_crops(c1,c2), 1 - _latitude_diff(c1,c2), _religion_sim(c1,c2)])



def acceptance(country,t):
    
    with open('data/acceptance_data/acceptance_indicators_dict.pkl','rb') as file:
        indicator_dict = pickle.load(file)
    
    def _neighbor_acceptance1(country,t):
        return indicator_dict[country]['neighbor1']['coeff'] * t + indicator_dict[country]['neighbor1']['intercept']
    
    def _neighbor_acceptance2(country,t):
        return indicator_dict[country]['neighbor2']['coeff'] * t + indicator_dict[country]['neighbor2']['intercept']
    
    def _job_scarce(country,t):
        return indicator_dict[country]['job_scarce']['coeff'] * t + indicator_dict[country]['job_scarce']['intercept']
    
    acceptance = np.average([_neighbor_acceptance1(country,t), _neighbor_acceptance2(country,t), _job_scarce(country,t)])
    
    if acceptance > 1:
        return 1
    elif acceptance < 0:
        return 0
    else: return acceptance
    

    
def CHPI(c1,c2,t,alpha,beta):
    
    return (alpha*sim(c1,c2) + beta*acceptance(c2,t)) / (alpha + beta)    



def viability(country):

    a = .3
    b = .3
    c = .2
    d = .2

    viability_features_df = pd.read_csv('data/viability_features.csv').set_index('Country')
    country = viability_features_df.loc[country]
    
    return a*(country["normGDP"]) + b*(1-country['normPopDens'])*(country['normPop']) + c*(country['normEmp']) + d*(country['Vulnerability score to climate change(ND-Gain index) in 2017'])