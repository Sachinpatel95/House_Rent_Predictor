from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd

boston_dataset=load_boston()

data=pd.DataFrame(data=boston_dataset.data,columns=boston_dataset.feature_names)

log_prices=np.log(boston_dataset.target)
target=pd.DataFrame(data=log_prices,columns=['PRICE'])
features=data.drop(['INDUS','AGE'],axis=1)
property_stats=features.mean().values.reshape(1,11)

from math import sqrt
regr=LinearRegression().fit(features,target)

fittedValues=regr.predict(features)

mse=mean_squared_error(target,fittedValues)
rmse=sqrt(mse)


def get_log_estimate(nr_rooms, students_per_room, next_to_river=False, high_confidence=True):
    property_stats[0][4]=nr_rooms
    property_stats[0][8]=students_per_room
    if(next_to_river):
        property_stats[0][2]=1

    else:
        property_stats[0][2]=0
    log_estimate=regr.predict(property_stats)[0][0]

    if(high_confidence):
        upper_bound=log_estimate+2*rmse
        lower_bound=log_estimate-2*rmse
        interval=95
    else:
        upper_bound=log_estimate+rmse
        lower_bound=log_estimate-rmse
        interval=68

    return log_estimate,upper_bound,lower_bound,interval

ZILLOW_MEDIAN_PRICE=583.3

scale_factor=ZILLOW_MEDIAN_PRICE/np.median(boston_dataset.target)

def get_dollar_estimate(RM,PTRATIO,CHAS=True,HIGH_CONFIDENCE=True):

    log_est,up_est,lower_est,confidence=get_log_estimate(RM,PTRATIO,next_to_river=CHAS,high_confidence=HIGH_CONFIDENCE)

    dollar_est=np.e**log_est*1000*scale_factor
    dollar_lower=np.e**lower_est*1000*scale_factor
    dollar_up=np.e**up_est*1000*scale_factor

    rounded_est=np.around(dollar_est,-3)
    rounded_high=np.around(dollar_up,-3)
    rounded_low=np.around(dollar_lower,-3)

    print(f"The estimated property value is USD {rounded_est}")
    print(f"At confidence {confidence}%")
    print(f"USD {rounded_low} at the lower end  to {rounded_high} at the higher end")

    return rounded_est,rounded_high,rounded_low,confidence