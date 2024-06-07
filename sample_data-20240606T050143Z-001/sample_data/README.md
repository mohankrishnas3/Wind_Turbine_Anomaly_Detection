## Table of Contents
1. [The Data](#the-data)
   * [Event Data](#event-data)
   * [Event Information](#event-information)
   * [Feature Description](#feature-description)

## The Data

The data consists of 95 datasets, containing 89 years of SCADA time series distributed across 36 different wind turbines
from the three wind farms A, B and C. The data for Wind farm A is based on data from the EDP-open data platform (https://www.edp.com/en/innovation/open-data/data), 
and consists of 5 wind turbines of an onshore wind farm in Portugal. 
It contains SCADA data and information derived by a given fault logbook which defines start timestamps for specified faults. 
From this data 22 datasets were selected to be included in this data collection. 
The other two wind farms are offshore wind farms located in Germany. All three datasets were anonymized due to confidentiality reasons for the wind farms A and B.
The overall dataset is balanced, as 44 out the 95 datasets contain a labeled anomaly event and the other 51 datasets represent normal behavior.
Each dataset is provided in form of a csv-file with columns defining the features and rows representing the data points of the time series.
Each of the wind farms ``A``,  ``B`` and ``C`` has its own directory where the file structure is given by:
* Wind Farm `<x>`
  * datasets
    * [<event_id>.csv](#event-data)
    * ...
    * [<event_id>.csv](#event-data)
  * [event_info.csv](#event-information)
  * [feature_description.csv](#feature-description)

### Event Data

The directory ``datasets`` contains datasets and every dataset contains one event, which is to be predicted.
Each dataset csv-file contains time series of one wind turbine, containing both training and prediction data.
The datasets are all high dimensional with 86, 257 or 957 features (depending on the wind farm).
The prediction data contains exactly one event which can either be an anomaly event (leading up to a failure) or just normal behavior (normal event).
The features are given in 10-minute average sensor measurements and for some sensors additional information in the form of 10-minute minimum, 
maximum and standard deviation of the sensor measurements are available. The datasets also contain columns for the status ID (``status_type_id``), 
the asset ID (``asset ID``), timestamp (``time_stamp``) and a row id (``id``).
The sensor data and time stamps are anonymized.

Each dataset file is a csv-file containing the following columns:
* ``time_stamp``: Time stamps in 10 minute frequency. Time stamps have been anonymized in order to not reveal the identity of the wind farms.
* ``asset_id``: Wind turbine ID. This can be used to find out which events took place on the same wind turbine. The IDs have also been anonymized.
* ``id``: An additional ID for each time stamp. This is supposed to prevent confusion with different time formats.
* ``train_test``: Strings describing whether the timestamp is part of the prediction data or the training data of the dataset.
* ``status_type``: Status of the wind turbine at the given time stamp. There are 6 different status IDs where status 0 and 2 are considered normal operation:
  * 0: Normal Operation - The turbine is in normal power production mode 
  * 1: Derated Operation - Derated power generation with a power restriction
  * 2: Idling - Asset is idling and waits to operate again
  * 3: Service - Asset is in service mode / service team is at the site
  * 4: Downtime - Asset is down due a fault or other reasons
  * 5: Other - Other operational states
* sensor_x_avg: 10 minute average of sensor measurements
* sensor_x_min: 10 minute minimum of sensor measurements (if available)
* sensor_x_max: 10 minute maximum of sensor measurements (if available)
* sensor_x_std: 10 minute standard deviation of sensor measurements (if available)

### Event Information

For each wind farm there is an event information csv-file which gives additional information for all events within the
``datasets`` directory. This file contains the following columns:
* ``event_id``: ID of the event
* ``event_label``: String describing whether the event of the dataset is an anomaly event (``anomaly``) or a normal event (``normal``)
* ``event_start``: Start time stamp of the event
* ``event_end``: End time stamp of the event
* ``event_start_id``: ID of the start time stamp of the event
* ``event_end_id``: ID of the end time stamp of the event
* ``event_description``: Additional information of the root cause for some anomaly events.

### Feature Description

The different features of each wind farm are described in the feature_description csv-files within the wind farm directories. These files contain the columns:
* ``sensor_name``: Anonymized name of the sensor.
* ``statistic_type``: String describing whether the sensor measurements are given as 10 minute ``average``, ``minimum``, ``maximum`` and/or ``standard deviation``.
* ``description``: Brief description of the sensor measurement.
* ``unit``: Unit of the given Sensor.
* ``is_angle``: Boolean value indicating whether the sensor represents an angle or not.
* ``is_counter``: Boolean value indicating whether the sensor represents a counter or not.
