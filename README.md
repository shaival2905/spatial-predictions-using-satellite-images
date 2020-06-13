# Spatial Predictions using Satellite Images

This repository contains the code and files for Spatial and Temporal Data Mining Project titled "Spatial Predictions using Satellite Images".

## Dataset Used:
Defense Science and Technology Laboratory (DSTL) has provided 1km x 1km satellite images in both 3 band and 16 band formats. It can be downloaded from [here](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data).<br>
The dataset has been labelled into 10 different classes:<br>
•	Buildings<br>
•	Small Vehicle<br>
•	Road<br>
•	Large Vehicle<br>
•	Track<br>
•	Miscellaneous Structures<br>
•	Trees<br>
•	Waterway<br>
•	Standing Water<br>
•	Crops<br>

|Type	|Value|
|:-------------:|:-------------|
|Sensor	|WorldView3|
|Wavebands	|•	Panchromatic: 450-800 nm <br>•	8 Multispectral: (red, red edge, coastal, blue, green, yellow, near-IR1 and near-IR2) 400 nm - 1040 nm<br>•	8 SWIR: 1195 nm - 2365 nm|
|Sensor Resolution (GSD) at Nadir|	•	Panchromatic: 0.31m <br>•	Multispectral: 1.24 m<br>•	SWIR: Delivered at 7.5m|
|Dynamic Range|•	Panchromatic and multispectral : 11-bits per pixel<br>•	SWIR : 14-bits per pixel|


## Libraries Used:
•	TensorFlow<br>
•	Numpy<br>
•	OpenCV<br>
•	Pandas<br>
•	Shapely<br>
•	Os<br>
•	Keras<br>
•	CSV<br>
•	Tifffle<br>
•	Skimage<br>
•	sys<br>

## Execution Instructions:
Run train.py for training.

