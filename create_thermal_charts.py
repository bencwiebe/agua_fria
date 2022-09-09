#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 09:58:13 2022

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
import pickle

#takes data_dict and the desired percentile to be analyzed, outputs a series of timeseries bar charts showing population leaf temperatures from block2 immediately next to block3,
#separated by population.

def plot_chart_by_pop(df, percentile):
    
    big_df = df
    
    big_df = big_df[big_df["plot"] != "LBW"] #LBW has some issues... shade, sometimes includes invasive plant that seems to be significantly cooler
    big_df["date_col"] = pd.to_datetime(big_df["date_col"], yearfirst = True)
    
    if percentile == 20:
        percentile_row = "20th_percentile"
    elif percentile == 50:
        percentile_row = "50th_percentile"
    elif percentile == 80:
        percentile_row = "80th_percentile"
    elif percentile == 95:
        percentile_row = "95th_percentile"
    else:
        pass
    
    nums = sorted(big_df.date_col.unique())
    conditions1 = [
      (big_df.date_col == nums[0]),
      (big_df.date_col == nums[1]),
      (big_df.date_col == nums[2]),
      (big_df.date_col == nums[3]),
      (big_df.date_col == nums[4]),
      (big_df.date_col == nums[5]),
      (big_df.date_col == nums[6]),
      (big_df.date_col == nums[7]),
      (big_df.date_col == nums[8])
    ]
    
    values = [1,2,3,4,5,6,7,8,9]
    big_df["Day"] = np.select(conditions1, values)
    
    big_df.index = big_df.Day
    
    block2 = big_df[big_df["block"] == "block2"]
    block3 = big_df[big_df["block"] == "block3"]
    
    ccr2 = block2[block2["plot"] == "CCR"]
    clf2 = block2[block2["plot"] == "CLF"]
    jla2 = block2[block2["plot"] == "JLA"]
    kkh2 = block2[block2["plot"] == "KKH"]
    lbw2 = block2[block2["plot"] == "LBW"]
    nrv2 = block2[block2["plot"] == "NRV"]
    
    ccr3 = block3[block3["plot"] == "CCR"]
    clf3 = block3[block3["plot"] == "CLF"]
    jla3 = block3[block3["plot"] == "JLA"]
    kkh3 = block3[block3["plot"] == "KKH"]
    lbw3 = block3[block3["plot"] == "LBW"]
    nrv3 = block3[block3["plot"] == "NRV"]
    
    
    days = [i for i in block2.Day.unique()]
    days = np.sort(days)
    
    block2_mean = block2.groupby(block2.Day)[percentile_row].mean()
    block3_mean = block3.groupby(block3.Day)[percentile_row].mean()
    
    ccr2_mean = ccr2.groupby(ccr2.Day)[percentile_row].mean()
    clf2_mean = clf2.groupby(clf2.Day)[percentile_row].mean()
    jla2_mean = jla2.groupby(jla2.Day)[percentile_row].mean()
    lbw2_mean = lbw2.groupby(lbw2.Day)[percentile_row].mean()
    nrv2_mean = nrv2.groupby(nrv2.Day)[percentile_row].mean()
    kkh2_mean = kkh2.groupby(kkh2.Day)[percentile_row].mean()
    
    ccr2_std = ccr2.groupby(ccr2.Day)[percentile_row].std()
    clf2_std = clf2.groupby(clf2.Day)[percentile_row].std()
    jla2_std = jla2.groupby(jla2.Day)[percentile_row].std()
    lbw2_std = lbw2.groupby(lbw2.Day)[percentile_row].std()
    nrv2_std = nrv2.groupby(nrv2.Day)[percentile_row].std()
    kkh2_std = kkh2.groupby(kkh2.Day)[percentile_row].std()
    
    ccr3_mean = ccr3.groupby(ccr3.Day)[percentile_row].mean()
    clf3_mean = clf3.groupby(clf3.Day)[percentile_row].mean()
    jla3_mean = jla3.groupby(jla3.Day)[percentile_row].mean()
    lbw3_mean = lbw3.groupby(lbw3.Day)[percentile_row].mean()
    nrv3_mean = nrv3.groupby(nrv3.Day)[percentile_row].mean()
    kkh3_mean = kkh3.groupby(kkh3.Day)[percentile_row].mean()
    
    ccr3_std = ccr3.groupby(ccr3.Day)[percentile_row].std()
    clf3_std = clf3.groupby(clf3.Day)[percentile_row].std()
    jla3_std = jla3.groupby(jla3.Day)[percentile_row].std()
    lbw3_std = lbw3.groupby(lbw3.Day)[percentile_row].std()
    nrv3_std = nrv3.groupby(nrv3.Day)[percentile_row].std()
    kkh3_std = kkh3.groupby(kkh3.Day)[percentile_row].std()
    
    x = np.arange(len(days))  # the label locations
    width = 0.1  # the width of the bars
    
    fig, ax = plt.subplots(4)
    fig.set_size_inches(18.5, 30)
    fig.suptitle("Variations in tree canopy temperature from block mean for each sampling date")
    
    
    ccr2_diff = (ccr2_mean - block2_mean)
    clf2_diff = (clf2_mean - block2_mean)
    jla2_diff = (jla2_mean - block2_mean)
    lbw2_diff = (lbw2_mean - block2_mean)
    nrv2_diff = (nrv2_mean - block2_mean)
    kkh2_diff = (kkh2_mean - block2_mean)
    ccr2_diff.dropna(axis = 0, inplace = True)
    clf2_diff.dropna(axis = 0, inplace = True)
    jla2_diff.dropna(axis = 0, inplace = True)
    lbw2_diff.dropna(axis = 0, inplace = True)
    nrv2_diff.dropna(axis = 0, inplace = True)
    kkh2_diff.dropna(axis = 0, inplace = True)
    
    ccr3_diff = (ccr3_mean - block3_mean)
    clf3_diff = (clf3_mean - block3_mean)
    jla3_diff = (jla3_mean - block3_mean)
    lbw3_diff = (lbw3_mean - block3_mean)
    nrv3_diff = (nrv3_mean - block3_mean)
    kkh3_diff = (kkh3_mean - block3_mean)
    ccr3_diff.dropna(axis = 0, inplace = True)
    clf3_diff.dropna(axis = 0, inplace = True)
    jla3_diff.dropna(axis = 0, inplace = True)
    lbw3_diff.dropna(axis = 0, inplace = True)
    nrv3_diff.dropna(axis = 0, inplace = True)
    kkh3_diff.dropna(axis = 0, inplace = True)
    
    ax[0].bar(ccr2_mean.index - 0.1, ccr2_diff, width, label='block2', yerr = ccr2_std, color = "red"   )
    ax[0].bar(ccr3_mean.index + 0.1, ccr3_diff, width, label='block3', yerr = ccr3_std, color = "green")
    ax[0].set_ylim(-5,5)
    ax[0].set_title("CCR")
    ax[0].legend()
    ax[0].set_ylabel("degrees C")
    ax[0].set_xlabel("Date")
    
    ax[1].bar(clf2_mean.index - 0.1, clf2_diff, width, label='block2', yerr = clf2_std, color = "red"   )
    ax[1].bar(clf3_mean.index + 0.1, clf3_diff, width, label='block3', yerr = clf3_std, color = "green")
    ax[1].set_ylim(-5,5)
    ax[1].set_title("CLF")
    ax[1].legend()
    ax[1].set_ylabel("degrees C")
    ax[1].set_xlabel("Date")
    
    ax[2].bar(nrv2_mean.index - 0.1, nrv2_diff, width, label='block2', yerr = nrv2_std, color = "red"   )
    ax[2].bar(nrv3_mean.index + 0.1, nrv3_diff, width, label='block3', yerr = nrv3_std, color = "green")
    ax[2].set_ylim(-5,5)
    ax[2].set_title("NRV")
    ax[2].legend()
    ax[2].set_ylabel("degrees C")
    ax[2].set_xlabel("Date")
    
    ax[3].bar(kkh2_mean.index - 0.1, kkh2_diff, width, label='block2', yerr = kkh2_std, color = "red"   )
    ax[3].bar(kkh3_mean.index + 0.1, kkh3_diff, width, label='block3', yerr = kkh3_std, color = "green")
    ax[3].set_ylim(-5,5)
    ax[3].set_title("KKH")
    ax[3].legend()
    ax[3].set_ylabel("degrees C")
    ax[3].set_xlabel("Date")
    
    plt.show()
  

#takes data_dict and desired percentile of leaf temperature to be used, and outputs a time-series bar chart showing leaf temperature deviations from the date_block mean temperature, separated
#by blocks. "date_block" mean temperature means the mean temperature of all leaf temperature measurements on a given day in the block of interest.
#for percentile, choose 20, 50, 80, or 95

def plot_charts(df, percentile):
    
    big_df = df
    big_df = big_df[big_df["plot"] != "LBW"] #LBW has some issues... shade, sometimes includes invasive plant that seems to be significantly cooler
    big_df["date_col"] = pd.to_datetime(big_df["date_col"], yearfirst = True)
    
    if percentile == 20:
        percentile_row = "20th_percentile"
    elif percentile == 50:
        percentile_row = "50th_percentile"
    elif percentile == 80:
        percentile_row = "80th_percentile"
    elif percentile == 95:
        percentile_row = "95th_percentile"
    else:
        pass
    
    big_df.loc[big_df["plot"] == "KKH", "adaptation"] = "cool"
    big_df.loc[big_df["plot"] == "NRV", "adaptation"] = "warm"
    big_df.loc[big_df["plot"] == "CCR", "adaptation"] = "warm"
    big_df.loc[big_df["plot"] == "CLF", "adaptation"] = "cool"
    big_df.loc[big_df["plot"] == "JLA", "adaptation"] = "cool"
    big_df.loc[big_df["plot"] == "LBW", "adaptation"] = "warm"
    
    nums = sorted(big_df.date_col.unique())
    conditions1 = [
      (big_df.date_col == nums[0]),
      (big_df.date_col == nums[1]),
      (big_df.date_col == nums[2]),
      (big_df.date_col == nums[3]),
      (big_df.date_col == nums[4]),
      (big_df.date_col == nums[5]),
      (big_df.date_col == nums[6]),
      (big_df.date_col == nums[7]),
      (big_df.date_col == nums[8])
    ]
    
    values = [1,2,3,4,5,6,7,8,9]
    big_df["Day"] = np.select(conditions1, values)
    
    big_df.index = big_df.Day
    
    block2 = big_df[big_df["block"] == "block2"]
    block3 = big_df[big_df["block"] == "block3"]
    
    ccr2 = block2[block2["plot"] == "CCR"]
    clf2 = block2[block2["plot"] == "CLF"]
    jla2 = block2[block2["plot"] == "JLA"]
    kkh2 = block2[block2["plot"] == "KKH"]
    lbw2 = block2[block2["plot"] == "LBW"]
    nrv2 = block2[block2["plot"] == "NRV"]
    
    ccr3 = block3[block3["plot"] == "CCR"]
    clf3 = block3[block3["plot"] == "CLF"]
    jla3 = block3[block3["plot"] == "JLA"]
    kkh3 = block3[block3["plot"] == "KKH"]
    lbw3 = block3[block3["plot"] == "LBW"]
    nrv3 = block3[block3["plot"] == "NRV"]
    
    days = [i for i in block2.Day.unique()]
    days = np.sort(days)
    
    block2_mean = block2.groupby(block2.Day)[percentile_row].mean()
    block3_mean = block3.groupby(block3.Day)[percentile_row].mean()
    
    ccr2_mean = ccr2.groupby(ccr2.Day)[percentile_row].mean()
    clf2_mean = clf2.groupby(clf2.Day)[percentile_row].mean()
    jla2_mean = jla2.groupby(jla2.Day)[percentile_row].mean()
    lbw2_mean = lbw2.groupby(lbw2.Day)[percentile_row].mean()
    nrv2_mean = nrv2.groupby(nrv2.Day)[percentile_row].mean()
    kkh2_mean = kkh2.groupby(kkh2.Day)[percentile_row].mean()
    
    ccr2_std = ccr2.groupby(ccr2.Day)[percentile_row].std()
    clf2_std = clf2.groupby(clf2.Day)[percentile_row].std()
    jla2_std = jla2.groupby(jla2.Day)[percentile_row].std()
    lbw2_std = lbw2.groupby(lbw2.Day)[percentile_row].std()
    nrv2_std = nrv2.groupby(nrv2.Day)[percentile_row].std()
    kkh2_std = kkh2.groupby(kkh2.Day)[percentile_row].std()
    
    ccr3_mean = ccr3.groupby(ccr3.Day)[percentile_row].mean()
    clf3_mean = clf3.groupby(clf3.Day)[percentile_row].mean()
    jla3_mean = jla3.groupby(jla3.Day)[percentile_row].mean()
    lbw3_mean = lbw3.groupby(lbw3.Day)[percentile_row].mean()
    nrv3_mean = nrv3.groupby(nrv3.Day)[percentile_row].mean()
    kkh3_mean = kkh3.groupby(kkh3.Day)[percentile_row].mean()
    
    ccr3_std = ccr3.groupby(ccr3.Day)[percentile_row].std()
    clf3_std = clf3.groupby(clf3.Day)[percentile_row].std()
    jla3_std = jla3.groupby(jla3.Day)[percentile_row].std()
    lbw3_std = lbw3.groupby(lbw3.Day)[percentile_row].std()
    nrv3_std = nrv3.groupby(nrv3.Day)[percentile_row].std()
    kkh3_std = kkh3.groupby(kkh3.Day)[percentile_row].std()
    
    x = np.arange(len(days))  # the label locations
    width = 0.1  # the width of the bars
    
    fig, ax = plt.subplots(2)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle("Variations in tree canopy temperature from block mean for each sampling date")
    
    ccr2_diff = (ccr2_mean - block2_mean)
    clf2_diff = (clf2_mean - block2_mean)
    jla2_diff = (jla2_mean - block2_mean)
    lbw2_diff = (lbw2_mean - block2_mean)
    nrv2_diff = (nrv2_mean - block2_mean)
    kkh2_diff = (kkh2_mean - block2_mean)
    ccr2_diff.dropna(axis = 0, inplace = True)
    clf2_diff.dropna(axis = 0, inplace = True)
    jla2_diff.dropna(axis = 0, inplace = True)
    lbw2_diff.dropna(axis = 0, inplace = True)
    nrv2_diff.dropna(axis = 0, inplace = True)
    kkh2_diff.dropna(axis = 0, inplace = True)
    
    ax[0].bar(ccr2_mean.index + 0.2, ccr2_diff, width, label='CCR', yerr = ccr2_std, color = "red"   )
    ax[0].bar(clf2_mean.index + 0.1, clf2_diff, width, label='CLF', yerr = clf2_std, color = "orange")
    ax[0].bar(jla2_mean.index + 0.0, jla2_diff, width, label='JLA', yerr = jla2_std, color = "pink"  )
    ax[0].bar(lbw2_mean.index - 0.0, lbw2_diff, width, label='LBW', yerr = lbw2_std, color = "green" )
    ax[0].bar(nrv2_mean.index - 0.1, nrv2_diff, width, label='NRV', yerr = nrv2_std, color = "blue"  )
    ax[0].bar(kkh2_mean.index - 0.2, kkh2_diff, width, label='KKH', yerr = kkh2_std, color = "purple")
    ax[0].set_ylim(-5, 5)
    ax[0].set_title("Block 2")
    ax[0].legend()
    # ax[0].set_xticklabels(dates)
    ax[0].set_ylabel("degrees C")
    ax[0].set_xlabel("Date")
    
    ccr3_diff = (ccr3_mean - block3_mean)
    clf3_diff = (clf3_mean - block3_mean)
    jla3_diff = (jla3_mean - block3_mean)
    lbw3_diff = (lbw3_mean - block3_mean)
    nrv3_diff = (nrv3_mean - block3_mean)
    kkh3_diff = (kkh3_mean - block3_mean)
    ccr3_diff.dropna(axis = 0, inplace = True)
    clf3_diff.dropna(axis = 0, inplace = True)
    jla3_diff.dropna(axis = 0, inplace = True)
    lbw3_diff.dropna(axis = 0, inplace = True)
    nrv3_diff.dropna(axis = 0, inplace = True)
    kkh3_diff.dropna(axis = 0, inplace = True)
    
    ax[1].bar(ccr3_mean.index + 0.2, ccr3_diff, width, label='CCR', yerr = ccr3_std, color = "red"   )
    ax[1].bar(clf3_mean.index + 0.1, clf3_diff, width, label='CLF', yerr = clf3_std, color = "orange")
    ax[1].bar(jla3_mean.index + 0.0, jla3_diff, width, label='JLA', yerr = jla3_std, color = "pink"  )
    ax[1].bar(lbw3_mean.index - 0.0, lbw3_diff, width, label='LBW', yerr = lbw3_std, color = "green" )
    ax[1].bar(nrv3_mean.index - 0.1, nrv3_diff, width, label='NRV', yerr = nrv3_std, color = "blue"  )
    ax[1].bar(kkh3_mean.index - 0.2, kkh3_diff, width, label='KKH', yerr = kkh3_std, color = "purple")
    ax[1].set_ylim(-5, 5)
    ax[1].set_title("Block 3")
    ax[1].legend()
    # ax[1].set_xticklabels(dates)
    ax[1].set_ylabel("degrees C")
    ax[1].set_xlabel("Date")
    
    plt.show()

#this function does the same thing as plot_charts, but it groups the populations into "cool-adapted" and "warm-adapted" populations, based on whether the Mean Annual Temperature
# of the population source location falls above or below the Mean Annual Temperature of the Agua Fria common garden. This defines NRV, CCR, and LBW as warm-adapted, which matches what Blasini
# et al. 2022 defines as warm-adapted as well
#This function is identical to plot_charts_by_adaptation_median, except it uses mean() to average the temperature percentile value from each file to get the adaptation average

def plot_charts_by_adaptation(df, percentile):
    
    big_df = df
    
    if percentile == 20:
        percentile_row = "20th_percentile"
    elif percentile == 50:
        percentile_row = "50th_percentile"
    elif percentile == 80:
        percentile_row = "80th_percentile"
    elif percentile == 95:
        percentile_row = "95th_percentile"
    else:
        pass
    
    big_df = big_df[big_df["plot"] != "LBW"] #LBW has some issues... shade, sometimes includes invasive plant that seems to be significantly cooler
    big_df["date_col"] = pd.to_datetime(big_df["date_col"], yearfirst = True)
    
    big_df.loc[big_df["plot"] == "KKH", "adaptation"] = "cool"
    big_df.loc[big_df["plot"] == "NRV", "adaptation"] = "warm"
    big_df.loc[big_df["plot"] == "CCR", "adaptation"] = "warm"
    big_df.loc[big_df["plot"] == "CLF", "adaptation"] = "cool"
    big_df.loc[big_df["plot"] == "JLA", "adaptation"] = "cool"
    big_df.loc[big_df["plot"] == "LBW", "adaptation"] = "warm"
    
    nums = sorted(big_df.date_col.unique())
    conditions1 = [
      (big_df.date_col == nums[0]),
      (big_df.date_col == nums[1]),
      (big_df.date_col == nums[2]),
      (big_df.date_col == nums[3]),
      (big_df.date_col == nums[4]),
      (big_df.date_col == nums[5]),
      (big_df.date_col == nums[6]),
      (big_df.date_col == nums[7]),
      (big_df.date_col == nums[8])
    ]
    
    values = [1,2,3,4,5,6,7,8,9]
    big_df["Day"] = np.select(conditions1, values)
    
    big_df.index = big_df.Day
    
    block2 = big_df[big_df["block"] == "block2"]
    block3 = big_df[big_df["block"] == "block3"]
    
    warm2 = block2[block2["adaptation"] == "warm"]
    warm3 = block3[block3["adaptation"] == "warm"]
    cool2 = block2[block2["adaptation"] == "cool"]
    cool3 = block3[block3["adaptation"] == "cool"]
    
    days = [i for i in block2.Day.unique()]
    days = np.sort(days)
    
    overall_date_mean = big_df.groupby(big_df.Day)[percentile_row].mean()
    block2_mean = block2.groupby(block2.Day)[percentile_row].mean()
    block3_mean = block3.groupby(block3.Day)[percentile_row].mean()
    
    warm2_mean = warm2.groupby(warm2.Day)[percentile_row].mean()
    warm3_mean = warm3.groupby(warm3.Day)[percentile_row].mean()
    cool2_mean = cool2.groupby(cool2.Day)[percentile_row].mean()
    cool3_mean = cool3.groupby(cool3.Day)[percentile_row].mean()
    warm2_std = warm2.groupby(warm2.Day)[percentile_row].std()
    warm3_std = warm3.groupby(warm3.Day)[percentile_row].std()
    cool2_std = cool2.groupby(cool2.Day)[percentile_row].std()
    cool3_std = cool3.groupby(cool3.Day)[percentile_row].std()
    
    x = np.arange(len(days))  # the label locations
    width = 0.1  # the width of the bars
    
    fig, ax = plt.subplots(2)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle("Variations in tree canopy temperature from block mean for each sampling date -- using {}th percentile value from each image".format(percentile))
    warm2_diff = (warm2_mean - block2_mean)
    cool2_diff = (cool2_mean - block2_mean)
    warm2_diff.dropna(axis = 0, inplace = True)
    cool2_diff.dropna(axis = 0, inplace = True)
    
    ax[0].bar(warm2_diff.index + 0.2, warm2_diff, width, label='warm2', yerr = np.sqrt(warm2_std**2 + warm2_std**2), color = "red"   )
    ax[0].bar(cool2_diff.index + 0.1, cool2_diff, width, label='cool2', yerr = np.sqrt(cool2_std**2 + cool2_std**2), color = "green")
    ax[0].set_ylim(-5, 5)
    ax[0].set_title("Block 2")
    ax[0].legend()
    # ax[0].set_xticklabels(dates)
    ax[0].set_ylabel("degrees C")
    ax[0].set_xlabel("Date")
    
    warm3_diff = (warm3_mean - block3_mean)
    cool3_diff = (cool3_mean - block3_mean)
    warm3_diff.dropna(axis = 0, inplace = True)
    cool3_diff.dropna(axis = 0, inplace = True)
    
    ax[1].bar(warm3_mean.index + 0.2, warm3_diff, width, label='warm3', yerr = np.sqrt(warm3_std**2 + warm3_std**2), color = "red"   )
    ax[1].bar(cool3_mean.index + 0.1, cool3_diff, width, label='cool3', yerr = np.sqrt(cool3_std**2 + cool3_std**2), color = "green")
    ax[1].set_ylim(-5, 5)
    ax[1].set_title("Block 3")
    ax[1].legend()
    # ax[1].set_xticklabels(dates)
    ax[1].set_ylabel("degrees C")
    ax[1].set_xlabel("Date")

    
    plt.show()

#same as plot_charts_by_adaptation but uses median to combine the percentile temperatures into a group average (in this case based off of adaptation)
#for percentile, chose 20, 50, 80, or 95

def plot_charts_by_adaptation_median(df, percentile):
    
    if percentile == 20:
        percentile_row = "20th_percentile"
    elif percentile == 50:
        percentile_row = "50th_percentile"
    elif percentile == 80:
        percentile_row = "80th_percentile"
    elif percentile == 95:
        percentile_row = "95th_percentile"
    else:
        pass
    
    big_df = df
    
    # big_df['percentile_temp'] = big_df.apply(lambda row: np.percentile(row.value_array, percentile), axis=1)
    big_df = big_df[big_df["plot"] != "LBW"] #LBW has some issues... shade, sometimes includes invasive plant that seems to be significantly cooler
    big_df["date_col"] = pd.to_datetime(big_df["date_col"], yearfirst = True)
    
    big_df.loc[big_df["plot"] == "KKH", "adaptation"] = "cool"
    big_df.loc[big_df["plot"] == "NRV", "adaptation"] = "warm"
    big_df.loc[big_df["plot"] == "CCR", "adaptation"] = "warm"
    big_df.loc[big_df["plot"] == "CLF", "adaptation"] = "cool"
    big_df.loc[big_df["plot"] == "JLA", "adaptation"] = "cool"
    big_df.loc[big_df["plot"] == "LBW", "adaptation"] = "warm"
    
    nums = sorted(big_df.date_col.unique())
    conditions1 = [
      (big_df.date_col == nums[0]),
      (big_df.date_col == nums[1]),
      (big_df.date_col == nums[2]),
      (big_df.date_col == nums[3]),
      (big_df.date_col == nums[4]),
      (big_df.date_col == nums[5]),
      (big_df.date_col == nums[6]),
      (big_df.date_col == nums[7]),
      (big_df.date_col == nums[8])
    ]
    
    values = [1,2,3,4,5,6,7,8,9]
    big_df["Day"] = np.select(conditions1, values)
    
    big_df.index = big_df.Day
    
    block2 = big_df[big_df["block"] == "block2"]
    block3 = big_df[big_df["block"] == "block3"]
    
    warm2 = block2[block2["adaptation"] == "warm"]
    warm3 = block3[block3["adaptation"] == "warm"]
    cool2 = block2[block2["adaptation"] == "cool"]
    cool3 = block3[block3["adaptation"] == "cool"]
    
    days = [i for i in block2.Day.unique()]
    days = np.sort(days)
    
    overall_date_median = big_df.groupby(big_df.Day)[percentile_row].median()
    block2_median = block2.groupby(block2.Day)[percentile_row].median()
    block3_median = block3.groupby(block3.Day)[percentile_row].median()
    
    warm2_median = warm2.groupby(warm2.Day)[percentile_row].median()
    warm3_median = warm3.groupby(warm3.Day)[percentile_row].median()
    cool2_median = cool2.groupby(cool2.Day)[percentile_row].median()
    cool3_median = cool3.groupby(cool3.Day)[percentile_row].median()
    warm2_std = warm2.groupby(warm2.Day)[percentile_row].std()
    warm3_std = warm3.groupby(warm3.Day)[percentile_row].std()
    cool2_std = cool2.groupby(cool2.Day)[percentile_row].std()
    cool3_std = cool3.groupby(cool3.Day)[percentile_row].std()
    
    x = np.arange(len(days))  # the label locations
    width = 0.1  # the width of the bars
    
    fig, ax = plt.subplots(2)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle("Variations in tree canopy temperature from block mean for each sampling date -- using {}th percentile in each image".format(percentile))
    
    warm2_diff = (warm2_median - block2_median)
    cool2_diff = (cool2_median - block2_median)
    warm2_diff.dropna(axis = 0, inplace = True)
    cool2_diff.dropna(axis = 0, inplace = True)
    
    ax[0].bar(warm2_diff.index + 0.2, warm2_diff, width, label='warm2', yerr = np.sqrt(warm2_std**2 + warm2_std**2), color = "red"   )
    ax[0].bar(cool2_diff.index + 0.1, cool2_diff, width, label='cool2', yerr = np.sqrt(cool2_std**2 + cool2_std**2), color = "green")
    ax[0].set_ylim(-10, 10)
    ax[0].set_title("Block 2")
    ax[0].legend()
    # ax[0].set_xticklabels(dates)
    ax[0].set_ylabel("degrees C")
    ax[0].set_xlabel("Date")
    
    warm3_diff = (warm3_median - block3_median)
    cool3_diff = (cool3_median - block3_median)
    warm3_diff.dropna(axis = 0, inplace = True)
    cool3_diff.dropna(axis = 0, inplace = True)
    
    ax[1].bar(warm3_median.index + 0.2, warm3_diff, width, label='warm3', yerr = np.sqrt(warm3_std**2 + warm3_std**2), color = "red"   )
    ax[1].bar(cool3_median.index + 0.1, cool3_diff, width, label='cool3', yerr = np.sqrt(cool3_std**2 + cool3_std**2), color = "green")
    ax[1].set_ylim(-5, 5)
    ax[1].set_title("Block 3")
    ax[1].legend()
    ax[1].set_ylabel("degrees C")
    ax[1].set_xlabel("Date")
    
    plt.show()

### find percentage above certain temperature ####
def find_perc_above(df, tcrit):
    
    overall_warm_dist = []
    overall_cool_dist = []
    
    block2_cool_dist = []
    block2_warm_dist = []
    block3_cool_dist = []
    block3_warm_dist = []
    
    cool2_abs_temp = []
    warm2_abs_temp = []
    cool3_abs_temp = []
    warm3_abs_temp = []
    
    # for i, row in df.iterrows():
    #     if df.loc
    # 	print(f"Index: {i}")
    # 	print(f"{row['0']}")
    
    cool2_abs_temp = np.array(cool2_abs_temp)
    warm2_abs_temp = np.array(warm2_abs_temp)
    cool3_abs_temp = np.array(cool3_abs_temp)
    warm3_abs_temp = np.array(warm3_abs_temp)
    
    tcrit = tcrit
    
    cool2_bool = cool2_abs_temp > tcrit
    warm2_bool = warm2_abs_temp > tcrit
    cool3_bool = cool3_abs_temp > tcrit
    warm3_bool = warm3_abs_temp > tcrit
    
    # count = np.count_nonzero(arr)
    cool2_perc_above = (np.count_nonzero(cool2_bool) / len(cool2_abs_temp) * 100)
    warm2_perc_above = (np.count_nonzero(warm2_bool) / len(warm2_abs_temp) * 100)
    cool3_perc_above = (np.count_nonzero(cool3_bool) / len(cool3_abs_temp) * 100)
    warm3_perc_above = (np.count_nonzero(warm3_bool) / len(warm3_abs_temp) * 100)
    
    print("percentage of Block 2 cool-adapted leaves above {}ºC: {:.2f}%".format(tcrit, cool2_perc_above))
    print("percentage of Block 2 warm-adapted leaves above {}ºC: {:.2f}%".format(tcrit, warm2_perc_above))
    print("percentage of Block 3 cool-adapted leaves above {}ºC: {:.2f}%".format(tcrit, cool3_perc_above))
    print("percentage of Block 3 warm-adapted leaves above {}ºC: {:.2f}%".format(tcrit, warm3_perc_above))
    
    data = [cool2_perc_above, warm2_perc_above, cool3_perc_above, warm3_perc_above]
    plt.bar(x = ['cool2', 'warm2', 'cool3', 'warm3'], height = data)
    plt.show()

########### MAIN #############

df = pd.read_csv("dataframe")

# data_dict=pickle.load(open('data_dict.pkl','rb'))
with open('data_dict.pkl', 'rb') as f:
    data_dict = pickle.load(f)

percentile = 95

plot_chart_by_pop(df, percentile)
plot_charts(df, percentile)
plot_charts_by_adaptation(df, percentile)
plot_charts_by_adaptation_median(df, percentile)













