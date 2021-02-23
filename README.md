# Air-passengers-prediction

## Introduction

The data set was donated to us by an unnamed company handling flight ticket reservations. The data is thin, it contains

* the date of departure
* the departure airport
* the arrival airport
* the mean and standard deviation of the number of weeks of the reservations made before the departure date
* a field called log_PAX which is related to the number of passengers (the actual number were changed for privacy reasons)

**The goal is to predict the log_PAX column. The prediction quality is measured by RMSE.**

**The data is obviously limited, but since data and location informations are available, it can be joined to external data sets. The challenge in this RAMP is to find good data that can be correlated to flight traffic.**

## Original data
<p align="center">
  <img src="https://github.com/Mehdi2402/images/blob/main/deep_initial_data.PNG?raw=true" />
</p>

## External data added
<p align="center">
  <img src="https://github.com/Mehdi2402/images/blob/main/deep_external_data.PNG?raw=true" />
</p>

## Results
* **RMSE : 0.286**

**Shap results to assess the value of external data**
<p align="center">
  <img src="https://github.com/Mehdi2402/images/blob/main/deep_shap1.png?raw=true" />
</p>

<p align="center">
  <img src="https://github.com/Mehdi2402/images/blob/main/deep_shap2.PNG?raw=true" />
</p>
