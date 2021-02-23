import math
import os
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder , OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def label_encoding(X): 
    mapper_ = {'day': {'31': 0,
      '2': 1,
      '3': 2,
      '4': 3,
      '1': 4,
      '29': 5,
      '5': 6,
      '8': 7,
      '24': 8,
      '25': 9,
      '26': 10,
      '28': 11,
      '23': 12,
      '22': 13,
      '19': 14,
      '30': 15,
      '21': 16,
      '27': 17,
      '7': 18,
      '15': 19,
      '18': 20,
      '12': 21,
      '17': 22,
      '9': 23,
      '14': 24,
      '10': 25,
      '11': 26,
      '6': 27,
      '16': 28,
      '20': 29,
      '13': 30},
     'weekday': {'5': 0, '6': 1, '1': 2, '4': 3, '2': 4, '0': 5, '3': 6},
     'week': {'52': 0,
      '47': 1,
      '27': 2,
      '1': 3,
      '51': 4,
      '10': 5,
      '14': 6,
      '35': 7,
      '4': 8,
      '5': 9,
      '7': 10,
      '44': 11,
      '3': 12,
      '9': 13,
      '8': 14,
      '2': 15,
      '6': 16,
      '33': 17,
      '31': 18,
      '48': 19,
      '50': 20,
      '32': 21,
      '36': 22,
      '13': 23,
      '34': 24,
      '28': 25,
      '11': 26,
      '45': 27,
      '49': 28,
      '15': 29,
      '21': 30,
      '30': 31,
      '12': 32,
      '19': 33,
      '39': 34,
      '24': 35,
      '26': 36,
      '46': 37,
      '17': 38,
      '40': 39,
      '22': 40,
      '29': 41,
      '16': 42,
      '25': 43,
      '43': 44,
      '37': 45,
      '41': 46,
      '18': 47,
      '23': 48,
      '42': 49,
      '38': 50,
      '20': 51},
     'month': {'12': 0,
      '1': 1,
      '2': 2,
      '3': 3,
      '11': 4,
      '7': 5,
      '8': 6,
      '4': 7,
      '9': 8,
      '10': 9,
      '6': 10,
      '5': 11},
     'year': {'2013': 0, '2012': 1, '2011': 2}}
    def label_encoder(X,col,mapper_):
        mapper = mapper_[col]
        X.loc[:,col] = X.loc[:,col].astype('str')
        for k in mapper.keys():
            X[col] = np.where(X[col]==k , mapper[k] , X[col] )
        X.loc[:,col] = X.loc[:,col].astype('float')
        return X

    for col in ['day','weekday','week','month','year'] :
        X = label_encoder(X,col,mapper_)
    return X

def _merge_external_data(X):
    filepath = os.path.join(
        os.path.dirname(__file__), 'external_data.csv'
    ) 
    X = X.copy()  # to avoid raising SettingOnCopyWarning
    # Make sure that DateOfDeparture is of dtype datetime
    X.loc[:, "DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture'])
    X.loc[:,'mois'] = X.loc[:, "DateOfDeparture"].dt.month
    X.loc[:,'annee'] = X.loc[:, "DateOfDeparture"].dt.year
    
    
    # Parse date to also be of dtype datetime
    external = pd.read_csv(filepath,sep=';')
    external.loc[:, "Date"] = pd.to_datetime(external['Date'])
    X_weather = external[['Date', 'AirPort', 'Max TemperatureC','Min TemperatureC']]
    X_weather = X_weather.rename(
        columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival'})

    X_merged = pd.merge(
        X, X_weather, how='left', on=['DateOfDeparture', 'Arrival'], sort=False
    )
    X_merged =  X_merged.rename(columns={'DateOfDeparture': 'Date'})


    static = external.drop(columns=['AirPort','Date','Max TemperatureC','Min TemperatureC','is_holiday','Dernier','passenger per year','plage','wage median','pax','annee','mois'])
    static = static.drop_duplicates()


    col_name_mapper = dict(zip(static.columns , [x + '_dep' for x in static.columns]))
    merged_dep = static.rename(columns=col_name_mapper)
    merged2 = pd.merge(
            X_merged, merged_dep, how='left', left_on='Departure',right_on = 'local_code_dep', sort=False
        )
    
    col_name_mapper = dict(zip(static.columns , [x + '_arr' for x in static.columns]))
    merged_arr = static.rename(columns=col_name_mapper)
    merged3 = pd.merge(
            merged2, merged_arr, how='left', left_on='Arrival',right_on = 'local_code_arr', sort=False
        )

    merged3.loc[:,['latitude_deg_arr','latitude_deg_dep','longitude_deg_arr',
               'longitude_deg_dep','score_dep','score_arr','scheduled_service_arr','scheduled_service_dep']] = merged3.loc[:,['latitude_deg_arr','latitude_deg_dep','longitude_deg_arr',
               'longitude_deg_dep','score_dep','score_arr','scheduled_service_arr','scheduled_service_dep']].astype('float')
    def haversine(coord1, coord2):

        R = 6372800  # Earth radius in meters
        lat1, lon1 = coord1
        lat2, lon2 = coord2

        phi1, phi2 = math.radians(lat1), math.radians(lat2) 
        dphi       = math.radians(lat2 - lat1)
        dlambda    = math.radians(lon2 - lon1)

        a = math.sin(dphi/2)**2 + \
            math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2

        return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))

    merged3['coord_dep'] = merged3.apply(lambda x : (x.latitude_deg_dep,x.longitude_deg_dep) , axis=1)
    merged3['coord_arr'] = merged3.apply(lambda x : (x.latitude_deg_arr,x.longitude_deg_arr) , axis=1)
    merged3['coord_dep'] = merged3['coord_dep'].apply(lambda x : tuple(x))
    merged3['coord_arr'] = merged3['coord_arr'].apply(lambda x : tuple(x))
    merged3['distance'] = merged3.apply(lambda x : haversine(x.coord_dep, x.coord_arr),axis=1)

    to_del = ['coord','local_code','longitude_deg','latitude_deg','city']                                                                                                                          

    merged4 = pd.merge(
            merged3, external[['Date','Dernier','is_holiday']].drop_duplicates() , how='left', on = 'Date', sort=False
        )

    merged4['Dernier'].fillna(method='bfill',inplace=True)
    merged4['Dernier'].fillna(method='ffill',inplace=True)
    
    select = ['AirPort','annee','mois','pax']
    
    merged = pd.merge(merged4, external[select].rename(columns={'AirPort':'Departure'}) , how='left' , on =['Departure','mois','annee'] ,sort=False)
    merged.rename(columns={'pax':'pax_dep'},inplace=True)
    merged.drop_duplicates(inplace=True)
    
    merged_last = pd.merge(merged, external[select].rename(columns={'AirPort':'Arrival'}) , how='left' , on =['Arrival','mois','annee'] ,sort=False)
    merged_last.drop_duplicates(inplace=True)
    merged_last.rename(columns={'pax':'pax_arr'},inplace=True)
    
    
    merged_last2 = pd.merge(merged_last, external[['AirPort','wage median','plage','passenger per year']].rename(columns={'AirPort':'Departure'}) , how='left' , on =['Departure'] ,sort=False)
    merged_last2.rename(columns={'wage median':'wage median_dep','plage':'plage_dep','passenger per year': 'passenger per year_dep'},inplace=True)
    merged_last2.drop_duplicates(inplace=True)
    
    merged_last22 = pd.merge(merged_last2, external[['AirPort','wage median','plage','passenger per year']].rename(columns={'AirPort':'Arrival'}) , how='left' , on =['Arrival'] ,sort=False)
    merged_last22.rename(columns={'wage median':'wage median_arr','plage':'plage_arr','passenger per year': 'passenger per year_arr'},inplace=True)
    merged_last22.drop_duplicates(inplace=True)
    
    return merged_last22.drop(columns=[x + '_arr' for x in to_del]+[x + '_dep' for x in to_del] + ['annee','mois'])



def _encode_dates(X):
    # With pandas < 1.0, we wil get a SettingWithCopyWarning
    # In our case, we will avoid this warning by triggering a copy
    # More information can be found at:
    # https://github.com/scikit-learn/scikit-learn/issues/16191

    X_encoded = X.copy()

    # Make sure that DateOfDeparture is of datetime format
    X_encoded.loc[:, 'Date'] = pd.to_datetime(X_encoded['Date'])
    # Encode the DateOfDeparture
    X_encoded.loc[:, 'year'] = X_encoded['Date'].dt.year
    X_encoded.loc[:, 'month'] = X_encoded['Date'].dt.month
    X_encoded.loc[:, 'day'] = X_encoded['Date'].dt.day
    X_encoded.loc[:, 'weekday'] = X_encoded['Date'].dt.weekday
    X_encoded.loc[:, 'week'] = X_encoded['Date'].dt.week
    X_encoded.loc[:, 'n_days'] = X_encoded['Date'].apply(
        lambda date: (date - pd.to_datetime("1970-01-01")).days
    )
    # Once we did the encoding, we will not need DateOfDeparture
    return X_encoded.drop(columns=["Date"])


def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates,validate=False)
    date_cols = ["Date"]

    scaler = StandardScaler()

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["Arrival", "Departure"]

    #categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    #categorical_cols = ["Arrival", "Departure"]

    data_merger = FunctionTransformer(_merge_external_data,validate=False)

    vars_to_encode = ['day','weekday','week','month','year']
    encode_labels = FunctionTransformer(label_encoding,validate=False)

    numerical_scaler = StandardScaler()
    numerical_cols = ["Max TemperatureC","Min TemperatureC","population_dep","population_arr",
                    "WeeksToDeparture", "std_wtd","score_dep","score_arr","Dernier"]


    preprocessor = make_column_transformer(
        (categorical_encoder, categorical_cols),
        remainder='passthrough'
        )

    xgb = GradientBoostingRegressor(loss='ls', learning_rate=0.1,
                n_estimators=2000, subsample=1.0, criterion='friedman_mse',
                min_samples_split=9, min_samples_leaf=5,
                min_weight_fraction_leaf=0.0, max_depth=4,
                min_impurity_decrease=0.0, min_impurity_split=None, random_state=123)


    return make_pipeline(data_merger, date_encoder, encode_labels, preprocessor, scaler,  xgb)
