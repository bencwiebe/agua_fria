#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 09:43:00 2022

@author: bcw269
"""

import pandas as pd
import os
import numpy as np
from PIL import Image
from numpy import asarray
from glob import glob
# import earthpy as et
# import earthpy.spatial as es
# import earthpy.plot as ep
# import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
# import plotly.graph_objects as go
# from plantcv import plantcv as pcv
from scipy import stats
from scipy.stats import kurtosis
from scipy.stats import skew

#function to create dictionary of dictionaries of folder structure. values = None
def f(path):
    if os.path.isdir(path):
        d = {}
        for name in os.listdir(path):
            if name == ".DS_Store":
                pass
            else:
                d[name] = f(os.path.join(path, name))
        return d
    else:
        pass

def create_data_dict(empty_dict):
    dir_list = [] #in case we ever need a list of subdirectories
    for date in empty_dict.keys():
      print(date)
      for block in empty_dict[date].keys():
        print(block)
        for plot in empty_dict[date][block].keys():
          print(plot)
          
          directory = dir + str(date) + "/" + str(block) + "/" + str(plot) +"/"
          dir_list.append(directory)
    
          #call both processing functions defined at beginning of notebook
          rgb_dict, thermal_dict, name_list = create_dict(directory)
          
          thermal_pixel_value_dict, pic_list = extract_leaf_thermal(rgb_dict, thermal_dict)
          
          data_dict = empty_dict.copy()
          
          #populate data_dict with thermal pixel values -- all values from each directory after masking for non-leaf pixels
          data_dict[date][block][plot] = thermal_pixel_value_dict
          
    return data_dict

def create_dict(dir): 

  #define dicts and lists for later
  rgb_dict = {}
  thermal_dict = {}
  name_list = []

  #iterate through files in a given folder (dir), extract rgb data from the jpg files, add them to rgb_dict with the number from
  #the file name as the key and the numpy ndarray as the value. Also adds name to the list name_list

  for file in os.listdir(dir):

    name = file[:-4]

    if file.endswith(".jpg"):

      image = Image.open(dir + file) #read jpg with PIL Image library
      data = np.array(image) #converts jpg to np.array

      rgb_dict[name] = data #write to dictionary

      name_list.append(name)

  #iterate through files in dir, read thermal csv files into pd.DataFrames, convert to numpy arrays, save to thermal_dict with name
  #as key

  for file in os.listdir(dir):
    name = file[:-4]
    if file.endswith(".csv"):

      read = pd.read_csv(dir + file, skiprows = 10)

      num_cols = len(read.axes[1])

      thermal = pd.read_csv(dir + file, skiprows = 10, usecols = range(1,num_cols), header = None)
      
      thermal_np = pd.DataFrame(thermal).to_numpy()
      thermal_dict[name] = thermal_np #write to dictionary
  
  return(rgb_dict, thermal_dict, name_list)

#outputs dictionary of files (keys) and thermal values (values) for each terminal directory

def extract_leaf_thermal(rgb_dict, thermal_dict):

  # Create masked image from a color image based RGB color-space and threshold values. 
  # for lower and upper_thresh list as: thresh = [red_thresh, green_thresh, blue_thresh]

  #initiate list to contain all thermal pixel values from trees within the population (mask applied)
  thermal_pixel_value_dict = {}

  #isolate file numbers, create list
  name_list = []
  for key in rgb_dict.keys():
    name_list.append(key)
  
  index_list = []

  #initiate list of images
  pic_list = []

  #iterate through each image pair, create and apply mask to thermal image, find desired percentile value, append that
  #value to temp_percentile_value list
  for name in name_list:
    
    #read in rgb and thermal data from dictionaries for each image
    rgb = rgb_dict[name]
    thermal = thermal_dict[name]

    #define the bands

    r =rgb[...,0]
    g = rgb[...,1]
    b = rgb[...,2]

    #mask based on survey by [Hamuda, Esmael; Glavin, Martin; Jones, Edward] --  Normalized Difference Index -- (G − R)/(G + R)
    mask = (g < r)
    #also get rid of shadows -- anything with a green pixel value of less than 100
    mask2 = g < 100
    mask3 = (g > 150) & (r > 150) & (b > 150)

    #get masked rgb by making copy, using BOTH filters
    masked_rgb = rgb.copy()
    masked_rgb[mask] = 0
    masked_rgb[mask2] = 0
    masked_rgb[mask3] = 0

    pic_list.append(rgb)
    pic_list.append(masked_rgb)

    #get masked thermal image by making copy, applying BOTH filters
    masked_therm = thermal.copy()
    masked_therm[mask] = 0
    masked_therm[mask2] = 0
    masked_therm[mask3] = 0

    #convert the 2d array to a 1d array... esentially a list of values
    thermal1d = masked_therm.ravel()

    #exclude the "masked" pixels, which have values of 0
    thermal1d = thermal1d[thermal1d != 0]

    if len(thermal1d) > 0:
      thermal_pixel_value_dict[name] = thermal1d
  
  return(thermal_pixel_value_dict, pic_list)

def create_df_with_adaptation(data_dict, image_percentile):

    rows = []
    for date in data_dict.keys():
      for block in data_dict[date].keys():
        for plot in data_dict[date][block].keys():
          for file1 in data_dict[date][block][plot].keys():
            value_array = data_dict[date][block][plot][file1]
            row = [date, block, plot, file1, value_array]
            rows.append(row)
    df = pd.DataFrame(rows, columns = ["date_col", "block", "plot", "file", "value_array"])
    
    df['percentile_temp'] = df.apply(lambda row: np.percentile(row.value_array, 50), axis=1)
    
    nums = sorted(df.date_col.unique())
    conditions1 = [
      (df.date_col == nums[0]),
      (df.date_col == nums[1]),
      (df.date_col == nums[2]),
      (df.date_col == nums[3]),
      (df.date_col == nums[4]),
      (df.date_col == nums[5]),
      (df.date_col == nums[6]),
      (df.date_col == nums[7]),
      (df.date_col == nums[8])
    ]
    
    values = [1,2,3,4,5,6,7,8,9]
    df["day_number"] = np.select(conditions1, values)
    
    df.loc[df["plot"] == "KKH", "adaptation"] = "cool"
    df.loc[df["plot"] == "NRV", "adaptation"] = "warm"
    df.loc[df["plot"] == "CCR", "adaptation"] = "warm"
    df.loc[df["plot"] == "CLF", "adaptation"] = "cool"
    df.loc[df["plot"] == "JLA", "adaptation"] = "cool"
    df.loc[df["plot"] == "LBW", "adaptation"] = "warm"
    
    df['indentifier'] = df.apply(lambda row: str(row.block)+str(row.adaptation), axis=1)
    
    return df


###### MAIN #######

dir = '8_26_22_delete_leafless/'

empty_dict = f(dir)

data_dict = create_data_dict(empty_dict)

df = create_df_with_adaptation(data_dict, 50)

df.to_csv("dataframe", index = False)







