#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 17:55:09 2022

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
import scipy.stats as stats

with open('data_dict.pkl', 'rb') as f:
    data_dict = pickle.load(f)


# Goals: end up with distributions by 1) overall block - date mean, 2) block and adaptation - date mean 2b) block and adaptation - block mean
# 3) overall adaptation - date mean 4) overall adaptation - date_block mean,

# everything_dict = {'200902': None, '200909': None, '200923': None, '201002': None,
#                    '201007': None, '201014': None, "201027": None, "201104": None, '201111': None}
# block_dict = {"block2": {"JLA": None, "CCR": None,  "CLF": None , "KKH": None, "LBW": None, "NRV": None, "JLV": None}, "block3": {"JLA": None, "CCR": None,  "CLF": None , "KKH": None, "LBW": None, "NRV": None, "JLV": None}}

date_dict = {'200902': None, '200909': None, '200923': None, '201002': None,
             '201007': None, '201014': None, "201027": None, "201104": None, '201111': None}
date_block_dict = {'200902': {"block2": None, "block3": None}, '200909': {"block2": None, "block3": None}, '200923': {"block2": None, "block3": None}, '201002': {"block2": None, "block3": None}, '201007': {
    "block2": None, "block3": None}, '201014': {"block2": None, "block3": None}, "201027": {"block2": None, "block3": None}, "201104": {"block2": None, "block3": None}, '201111': {"block2": None, "block3": None}}

# plot_dict = {"JLA": None, "CCR": None,  "CLF": None , "KKH": None, "LBW": None, "NRV": None, "JLV": None}
# name_list = ["JLA","CCR","CLF","KKH","LBW","NRV"]
# date_list = ['200902','200909','200923','201002','201007','201014',"201027","201104",'201111']

warm_cool_dict = {'200902': {"block2": {"cool": None, "warm": None}, "block3": {"cool": None, "warm": None}}, '200909': {"block2": {"cool": None, "warm": None}, "block3": {"cool": None, "warm": None}}, '200923': {"block2": {"cool": None, "warm": None}, "block3": {"cool": None, "warm": None}}, '201002': {"block2": {"cool": None, "warm": None}, "block3": {"cool": None, "warm": None}}, '201007': {"block2": {
    "cool": None, "warm": None}, "block3": {"cool": None, "warm": None}}, '201014': {"block2": {"cool": None, "warm": None}, "block3": {"cool": None, "warm": None}}, "201027": {"block2": {"cool": None, "warm": None}, "block3": {"cool": None, "warm": None}}, "201104": {"block2": {"cool": None, "warm": None}, "block3": {"cool": None, "warm": None}}, '201111': {"block2": {"cool": None, "warm": None}, "block3": {"cool": None, "warm": None}}}

# hist_dict = date_dict.copy()

# for date in everything_dict.keys():
#     everything_dict[date] = block_dict

# 1)
overall_block2_dist_minus_date_mean = []
overall_block3_dist_minus_date_mean = []

# 2)
overall_warm_dist_minus_date_mean = []
overall_cool_dist_minus_date_mean = []

# 2b
overall_warm_dist_minus_dateblock_mean = []
overall_cool_dist_minus_dateblock_mean = []

# 3
block2_cool_dist_minus_date_mean = []
block2_warm_dist_minus_date_mean = []
block3_cool_dist_minus_date_mean = []
block3_warm_dist_minus_date_mean = []

# 4
block2_cool_dist_minus_dateblock_mean = []
block2_warm_dist_minus_dateblock_mean = []
block3_cool_dist_minus_dateblock_mean = []
block3_warm_dist_minus_dateblock_mean = []

# 5
cool2_abs_temp_dist = []
warm2_abs_temp_dist = []
cool3_abs_temp_dist = []
warm3_abs_temp_dist = []

# fill in date_dict and date_block_dict for the purpose of having something to subtract later on
for date in data_dict.keys():
    date_dist = []
    for block in data_dict[date].keys():
        date_block_dist = []
        for plot in data_dict[date][block].keys():
            for file1 in data_dict[date][block][plot].keys():
                for num in data_dict[date][block][plot][file1]:
                    date_dist.append(num)
                    date_block_dist.append(num)
        date_block_mean = np.mean(date_block_dist)
        date_block_dict[date][block] = date_block_mean
    date_mean = np.mean(date_dist)
    date_dict[date] = date_mean


for date in data_dict.keys():

    # date_dist = []
    # block2_dist = []
    # block3_dist = []

    for block in data_dict[date].keys():

        block2_dist = []
        block3_dist = []
        warm_dist = []
        cool_dist = []

        for plot in data_dict[date][block].keys():

            # plot_dist = []

            if plot == "KKH":
                adaptation = "cool"
            elif plot == "NRV":
                adaptation = "warm"
            elif plot == "CCR":
                adaptation = "warm"
            elif plot == "CLF":
                adaptation = "cool"
            elif plot == "JLA":
                adaptation = "cool"
            elif plot == "LBW":
                adaptation = "warm"
            else:
                print("check out the if elif loop... we have a {} in date {} block {}".format(
                    plot, date, block))

            for file1 in data_dict[date][block][plot].keys():
                for num in data_dict[date][block][plot][file1]:

                    if block == "block2":
                        block2_dist.append(num)
                    elif block == "block3":
                        block3_dist.append(num)

                    if adaptation == "warm":
                        warm_dist.append(num)
                    elif adaptation == "cool":
                        cool_dist.append(num)

        date_block_mean = date_block_dict[date][block]
        date_mean = date_dict[date]

        block2_np = np.array(block2_dist)
        block3_np = np.array(block3_dist)
        warm_dist_np = np.array(warm_dist)
        cool_dist_np = np.array(cool_dist)

        block2_norm_by_date = block2_np - date_mean
        block3_norm_by_date = block3_np - date_mean
        warm_norm_by_date = warm_dist_np - date_mean
        cool_norm_by_date = cool_dist_np - date_mean

        warm_norm_by_dateblock = warm_dist_np - date_block_mean
        cool_norm_by_dateblock = cool_dist_np - date_block_mean

        for num in warm_norm_by_dateblock:
            overall_warm_dist_minus_dateblock_mean.append(num)
        for num in cool_norm_by_dateblock:
            overall_cool_dist_minus_dateblock_mean.append(num)

        for num in warm_norm_by_date:
            overall_warm_dist_minus_date_mean.append(num)
        for num in cool_norm_by_date:
            overall_cool_dist_minus_date_mean.append(num)

        for num in block2_norm_by_date:
            overall_block2_dist_minus_date_mean.append(num)
        for num in block3_norm_by_date:
            overall_block3_dist_minus_date_mean.append(num)


        if block == "block2":

            for num in warm_norm_by_date:
                block2_warm_dist_minus_date_mean.append(num)
            for num in cool_norm_by_date:
                block2_cool_dist_minus_date_mean.append(num)

            for num in warm_dist:
                warm2_abs_temp_dist.append(num)
            for num in cool_dist:
                cool2_abs_temp_dist.append(num)

        if block == "block3":

            for num in warm_norm_by_date:
                block3_warm_dist_minus_date_mean.append(num)
            for num in cool_norm_by_date:
                block3_cool_dist_minus_date_mean.append(num)

            for num in warm_dist:
                warm3_abs_temp_dist.append(num)
            for num in cool_dist:
                cool3_abs_temp_dist.append(num)

        if block == "block2":

            for num in warm_norm_by_dateblock:
                block2_warm_dist_minus_dateblock_mean.append(num)
            for num in cool_norm_by_dateblock:
                block2_cool_dist_minus_dateblock_mean.append(num)

        if block == "block3":

            for num in warm_norm_by_dateblock:
                block3_warm_dist_minus_dateblock_mean.append(num)
            for num in cool_norm_by_dateblock:
                block3_cool_dist_minus_dateblock_mean.append(num)

        warm_cool_dict[date][block]["warm"] = warm_dist
        warm_cool_dict[date][block]["cool"] = cool_dist

    # if block == "block2":
    #     for num in block_dist:
    #         overall_block2_dist_minus_date_mean.append(num)
    # elif block == "block3":
    #     for num in block_dist:
    #         overall_block3_dist_minus_date_mean.append(num)
    # else:
    #     print("check the block if loop")

#MEANS AND KURTOSES

# 1)
overall_block2_dist_minus_date_mean__mean = np.mean(overall_block2_dist_minus_date_mean)
overall_block2_dist_minus_date_mean__kurtosis = kurtosis(overall_block2_dist_minus_date_mean)
overall_block3_dist_minus_date_mean__mean = np.mean(overall_block3_dist_minus_date_mean)
overall_block3_dist_minus_date_mean__kurtosis = kurtosis(overall_block3_dist_minus_date_mean)

# 2)
overall_warm_dist_minus_date_mean__mean = np.mean(overall_warm_dist_minus_date_mean)
overall_warm_dist_minus_date_mean__kurtosis = kurtosis(overall_warm_dist_minus_date_mean)
overall_cool_dist_minus_date_mean__mean = np.mean(overall_cool_dist_minus_date_mean)
overall_cool_dist_minus_date_mean__kurtosis = kurtosis(overall_cool_dist_minus_date_mean)

# 2b
overall_warm_dist_minus_dateblock_mean__mean = np.mean(overall_warm_dist_minus_dateblock_mean)
overall_warm_dist_minus_dateblock_mean__kurtosis = kurtosis(overall_warm_dist_minus_dateblock_mean)
overall_cool_dist_minus_dateblock_mean__mean = np.mean(overall_cool_dist_minus_dateblock_mean)
overall_cool_dist_minus_dateblock_mean__kurtosis = kurtosis(overall_cool_dist_minus_dateblock_mean)

# 3
block2_cool_dist_minus_date_mean__mean = np.mean(block2_cool_dist_minus_date_mean)
block2_cool_dist_minus_date_mean__kurtosis = kurtosis(block2_cool_dist_minus_date_mean)
block2_warm_dist_minus_date_mean__mean = np.mean(block2_warm_dist_minus_date_mean)
block2_warm_dist_minus_date_mean__kurtosis = kurtosis(block2_warm_dist_minus_date_mean)
block3_cool_dist_minus_date_mean__mean = np.mean(block3_cool_dist_minus_date_mean)
block3_cool_dist_minus_date_mean__kurtosis = kurtosis(block3_cool_dist_minus_date_mean)
block3_warm_dist_minus_date_mean__mean = np.mean(block3_warm_dist_minus_date_mean)
block3_warm_dist_minus_date_mean__kurtosis = kurtosis(block3_warm_dist_minus_date_mean)

# 4
block2_cool_dist_minus_dateblock_mean__mean = np.mean(block2_cool_dist_minus_dateblock_mean)
block2_cool_dist_minus_dateblock_mean__kurtosis = kurtosis(block2_cool_dist_minus_dateblock_mean)
block2_warm_dist_minus_dateblock_mean__mean = np.mean(block2_warm_dist_minus_dateblock_mean)
block2_warm_dist_minus_dateblock_mean__kurtosis = kurtosis(block2_warm_dist_minus_dateblock_mean)
block3_cool_dist_minus_dateblock_mean__mean = np.mean(block3_cool_dist_minus_dateblock_mean)
block3_cool_dist_minus_dateblock_mean__kurtosis = kurtosis(block3_cool_dist_minus_dateblock_mean)
block3_warm_dist_minus_dateblock_mean__mean = np.mean(block3_warm_dist_minus_dateblock_mean)
block3_warm_dist_minus_dateblock_mean__kurtosis = kurtosis(block3_warm_dist_minus_dateblock_mean)

# 5
# cool2_abs_temp_dist = []
# warm2_abs_temp_dist = []
# cool3_abs_temp_dist = []
# warm3_abs_temp_dist = []

abs2 = []
abs2.append(warm2_abs_temp_dist)
abs2.append(cool2_abs_temp_dist)
abs3 = []
abs3.append(warm3_abs_temp_dist)
abs3.append(cool3_abs_temp_dist)
abswarm = []
abswarm.append(warm2_abs_temp_dist)
abswarm.append(warm3_abs_temp_dist)
abscool = []
abscool.append(cool2_abs_temp_dist)
abscool.append(cool3_abs_temp_dist)

##################################################################################################
##################################################################################################  

'''BY ADAPTATION -- ONLY HIST '''

fig, ax1 = plt.subplots(1, 1, figsize = (10, 10))
ax1.hist(overall_warm_dist_minus_dateblock_mean, color="green", range=(-15, 22), density=True,
          stacked=True, alpha=0.5, bins=100, label="Warm-adapted")
ax1.hist(overall_cool_dist_minus_dateblock_mean, color="red", range=(-15, 22), density=True,
          stacked=True, alpha=0.5, bins=100, label="Cool-adapted")

ax1.legend(prop = {"size": 20})
ax1.text(-7,0.06, "Mean: {:.2f}ºC\nKurtosis: {:.2f}".format(overall_warm_dist_minus_dateblock_mean__mean,overall_warm_dist_minus_dateblock_mean__kurtosis),horizontalalignment='right', size = 18)
ax1.text(7,0.04, "Mean: {:.2f}ºC\nKurtosis: {:.2f}".format(overall_cool_dist_minus_dateblock_mean__mean,overall_cool_dist_minus_dateblock_mean__kurtosis),horizontalalignment='left', size = 18)
ax1.tick_params(axis='both', which='major', labelsize=15)

fig.suptitle('Leaf Temperature Distributions of warm- and \ncool-adapted P. fremontii populations', size = 24)
fig.supxlabel('Leaf temperature distribution around date mean (ºC)', size = 20)
fig.supylabel("Normalized Bin Counts", size = 20)

plt.tight_layout()

plt.savefig("charts/adaptation_norm_by_dateblock.png", dpi=300)

##################################################################################################
##################################################################################################  

'''BY ADAPTATION -- HIST AND BOXPLOT'''

fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (20,10))

ax0.boxplot([overall_warm_dist_minus_dateblock_mean, overall_cool_dist_minus_dateblock_mean], showfliers = False, labels = ["warm-adapted", "cool-adapted"], whis = (10,90), widths = 0.45)
ax0.legend()
ax0.set_ylabel("Leaf temperature distribution \naround treatment mean (ºC)", size = 24)
ax0.set_xlabel("Adaptation", size = 24)
ax0.tick_params(axis='both', which='major', labelsize=18)

ax1.hist(overall_warm_dist_minus_dateblock_mean, color="green", range=(-15, 22), density=True,
          stacked=True, alpha=0.5, bins=100, label="Warm-adapted")
ax1.hist(overall_cool_dist_minus_dateblock_mean, color="red", range=(-15, 22), density=True,
          stacked=True, alpha=0.5, bins=100, label="Cool-adapted")

ax1.legend(prop = {"size": 20})
ax1.text(-7,0.06, "Mean: {:.2f}ºC\nKurtosis: {:.2f}".format(overall_warm_dist_minus_dateblock_mean__mean,overall_warm_dist_minus_dateblock_mean__kurtosis),horizontalalignment='right', size = 18)
ax1.text(7,0.04, "Mean: {:.2f}ºC\nKurtosis: {:.2f}".format(overall_cool_dist_minus_dateblock_mean__mean,overall_cool_dist_minus_dateblock_mean__kurtosis),horizontalalignment='left', size = 18)
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.set_ylabel("Normalized bin counts", size = 24)
ax1.set_xlabel('Leaf temperature distribution around treatment mean (ºC)', size = 22)

fig.suptitle('Leaf Temperature Distributions of warm- and cool-adapted P. fremontii populations', size = 28)

plt.tight_layout()
plt.subplots_adjust(wspace=0.16)

plt.savefig("charts/adaptation_norm_by_dateblock.png", dpi=300)

plt.clf()


##################################################################################################
##################################################################################################  

''' BY WATERING TREATMENT ---- HIST AND BOXPLOT''' 

fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (20,10))

ax0.boxplot([overall_block3_dist_minus_date_mean, overall_block2_dist_minus_date_mean], showfliers = False, labels = ["Irrigated", "Droughted"], whis = (10,90), widths = 0.45)
ax0.legend()
ax0.set_ylabel("Leaf temperature distribution \naround sampling date mean (ºC)", size = 24)
ax0.set_xlabel("Irrigation treatment", size = 24)
ax0.tick_params(axis='both', which='major', labelsize=18)

ax1.hist(overall_block3_dist_minus_date_mean, color="green", range=(-12, 20), density=True,
          stacked=True, alpha=0.5, bins=100, label="Irrigated")
ax1.hist(overall_block2_dist_minus_date_mean, color="red", range=(-12, 20), density=True,
          stacked=True, alpha=0.5, bins=100, label="Droughted")
ax1.legend(prop = {"size": 20})
ax1.text(-5,0.09, "Mean: {:.2f}ºC\nKurtosis: {:.2f}".format(overall_block3_dist_minus_date_mean__mean,overall_block3_dist_minus_date_mean__kurtosis),horizontalalignment='right', size = 18)
ax1.text(5,0.04, "Mean: {:.2f}ºC\nKurtosis: {:.2f}".format(overall_block2_dist_minus_date_mean__mean,overall_block2_dist_minus_date_mean__kurtosis),horizontalalignment='left', size = 18)
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.set_ylabel("Normalized Bin Counts", size = 20)
ax1.set_xlabel('Leaf temperature distribution around sampling date mean (ºC)', size = 20)

fig.suptitle('Leaf Temperature Distributions of irrigated and droughted \nblocks of P. fremontii common garden', size = 24)

plt.tight_layout()

plt.savefig("charts/water_treatment_norm_by_date.png", dpi=300)

plt.clf()

##################################################################################################
##################################################################################################

'''' UNWATERED ONLY --  HIST BY ADAPTATION '''

plt.hist(block2_warm_dist_minus_dateblock_mean, color="green", range=(-20, 20), density=True,
          stacked=True, alpha=0.5, bins=100, label="Warm-adapted \npopulations")
plt.hist(block2_cool_dist_minus_dateblock_mean, color="red", range=(-20, 20), density=True,
          stacked=True, alpha=0.5, bins=100, label="Cool-adapted \npopulations")
plt.legend()
plt.ylim(0,0.18)
plt.title("Droughted P. fremontii leaf temperature distribution,\n by thermal adaptation")
plt.ylabel("Normalized Bin Counts")
plt.xlabel("Deviations in leaf pixel values from treatment mean (ºC)")

plt.savefig("charts/Unwatered_norm_dist_by_adaptation.png", dpi=300)

plt.show()

plt.clf()


##################################################################################################
##################################################################################################

'''' WATERED ONLY -- HIST BY ADAPTATION'''

plt.hist(block3_warm_dist_minus_dateblock_mean, color="green", range=(-20, 20), density=True,
          stacked=True, alpha=0.5, bins=100, label="Warm-adapted\npopulations")

plt.hist(block3_cool_dist_minus_dateblock_mean, color="red", range=(-20, 20), density=True,
          stacked=True, alpha=0.5, bins=100, label="Cool-adapted\npopulations")
plt.legend()
plt.ylim(0,0.18)
plt.title("Irrigated P. fremontii leaf temperature distribution,\n by thermal adaptation")
plt.ylabel("Normalized Bin Counts")
plt.xlabel("Deviations in leaf pixel values from treatment mean (ºC)")

plt.savefig("charts/watered_norm_dist_by_adaptation.png", dpi=300)

plt.show()

##################################################################################################
##################################################################################################  

'''' WARM-ADAPTED ONLY -- HIST BY BLOCK'''

plt.hist(block3_warm_dist_minus_dateblock_mean, color="green", range=(-20, 20), density=True,
          stacked=True, alpha=0.5, bins=100, label="Irrigated")

plt.hist(block2_warm_dist_minus_dateblock_mean, color="red", range=(-20, 20), density=True,
          stacked=True, alpha=0.5, bins=100, label="Droughted")
plt.legend()
# plt.ylim(0,0.18)
plt.title("Warm-adapted P. fremontii leaf temperature distributions by irrigation treatment")
plt.ylabel("Normalized Bin Counts")
plt.xlabel("Deviations in leaf pixel values from treatment mean (ºC)")

plt.savefig("charts/warm_adapted_norm_dist_by_treatmenet.png", dpi=300)

plt.show()

##################################################################################################
##################################################################################################  

''''COOL-ADAPTED ONLY -- HIST BY BLOCK'''

plt.hist(block3_cool_dist_minus_dateblock_mean, color="green", range=(-20, 20), density=True,
          stacked=True, alpha=0.5, bins=100, label="Irrigated")

plt.hist(block2_cool_dist_minus_dateblock_mean, color="red", range=(-20, 20), density=True,
          stacked=True, alpha=0.5, bins=100, label="Droughted")
plt.legend()
# plt.ylim(0,0.18)
plt.title("Cool-adapted P. fremontii leaf temperature distributions by irrigation treatment")
plt.ylabel("Normalized Bin Counts")
plt.xlabel("Deviations in leaf pixel values from treatment mean (ºC)")

plt.savefig("charts/cool_adapted_norm_dist_by_treatmenet.png", dpi=300)

plt.show()

plt.clf()
##################################################################################################
##################################################################################################  

'''BY TREATMENT AND BLOCK --- NORMALIZED HISTS ONLY -- DATEBLOCK-AVERAGED'''

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True,  figsize = (20, 10))

ax1.hist(block3_warm_dist_minus_dateblock_mean, color="green", range=(-20, 20), density=True,
          stacked=True, alpha=0.5, bins=100, label="Warm-adapted\npopulations")
ax1.hist(block3_cool_dist_minus_dateblock_mean, color="red", range=(-20, 20), density=True,
          stacked=True, alpha=0.5, bins=100, label="Cool-adapted\npopulations")
ax1.set_title('Irrigated', size = 20)
ax1.set_xlabel('Leaf temperature distribution around treatment mean (ºC)', size = 14)
ax1.legend(prop = {"size": 20})
ax1.text(-3.5,0.14, "Mean: {:.2f}ºC\nKurtosis: {:.2f}".format(block3_warm_dist_minus_dateblock_mean__mean,block3_warm_dist_minus_dateblock_mean__kurtosis),horizontalalignment='right', size = 18)
ax1.text(5,0.06, "Mean: {:.2f}ºC\nKurtosis: {:.2f}".format(block3_cool_dist_minus_dateblock_mean__mean,block3_cool_dist_minus_dateblock_mean__kurtosis),horizontalalignment='left', size = 18)


ax2.hist(block2_warm_dist_minus_dateblock_mean, color="green", range=(-20, 20), density=True,
          stacked=True, alpha=0.5, bins=100, label="Warm-adapted\npopulations")
ax2.hist(block2_cool_dist_minus_dateblock_mean, color="red", range=(-20, 20), density=True,
          stacked=True, alpha=0.5, bins=100, label="Cool-adapted\npopulations")
ax2.set_xlabel('Leaf temperature distribution around treatment mean (ºC)', size = 14)
ax2.set_title('Droughted', size = 20)
ax2.legend(prop = {"size": 20})
ax2.text(-7,0.06, "Mean: {:.2f}ºC\nKurtosis: {:.2f}".format(block2_warm_dist_minus_dateblock_mean__mean,block2_warm_dist_minus_dateblock_mean__kurtosis),horizontalalignment='right', size = 18)
ax2.text(7,0.04, "Mean: {:.2f}ºC\nKurtosis: {:.2f}".format(block2_cool_dist_minus_dateblock_mean__mean,block2_cool_dist_minus_dateblock_mean__kurtosis),horizontalalignment='left', size = 18)


fig.suptitle("P. fremontii leaf temperature distributions around block mean by treatment and adaptation", size = 30)
fig.supylabel('Normalized Bin Counts', size = 24)

plt.tight_layout()

plt.savefig("charts/treatment_and_adaptation_normalized_hist_dateblock_ave.png", dpi=300)

plt.show()

plt.clf()

##################################################################################################
##################################################################################################  

'''BY TREATMENT AND BLOCK --- NORMALIZED HISTS ONLY -- DATE-AVERAGED'''

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex = True, figsize = (20, 10))

ax1.hist(block3_warm_dist_minus_date_mean, color="green", range=(-20, 20), density=True,
          stacked=True, alpha=0.5, bins=100, label="Warm-adapted\npopulations")
ax1.hist(block3_cool_dist_minus_date_mean, color="red", range=(-20, 20), density=True,
          stacked=True, alpha=0.5, bins=100, label="Cool-adapted\npopulations")
ax1.set_title('Irrigated', size = 20)
# ax1.set_xlabel('Leaf temperature distribution around treatment mean (ºC)', size = 24)
ax1.legend(prop = {"size": 20})
ax1.text(-3.5,0.14, "Mean: {:.2f}ºC\nKurtosis: {:.2f}".format(block3_warm_dist_minus_date_mean__mean,block3_warm_dist_minus_date_mean__kurtosis),horizontalalignment='right', size = 18)
ax1.text(5,0.06, "Mean: {:.2f}ºC\nKurtosis: {:.2f}".format(block3_cool_dist_minus_date_mean__mean,block3_cool_dist_minus_date_mean__kurtosis),horizontalalignment='left', size = 18)
ax1.tick_params(axis='both', which='major', labelsize=15)

ax2.hist(block2_warm_dist_minus_date_mean, color="green", range=(-20, 20), density=True,
          stacked=True, alpha=0.5, bins=100, label="Warm-adapted\npopulations")
ax2.hist(block2_cool_dist_minus_date_mean, color="red", range=(-20, 20), density=True,
          stacked=True, alpha=0.5, bins=100, label="Cool-adapted\npopulations")
# ax2.set_xlabel('Leaf temperature distribution around treatment mean (ºC)', size = 24)
ax2.set_title('Droughted', size = 20)
ax2.legend(prop = {"size": 20})
ax2.text(-7,0.06, "Mean: {:.2f}ºC\nKurtosis: {:.2f}".format(block2_warm_dist_minus_date_mean__mean,block2_warm_dist_minus_date_mean__kurtosis),horizontalalignment='right', size = 18)
ax2.text(7,0.04, "Mean: {:.2f}ºC\nKurtosis: {:.2f}".format(block2_cool_dist_minus_date_mean__mean,block2_cool_dist_minus_date_mean__kurtosis),horizontalalignment='left', size = 18)
ax2.tick_params(axis='both', which='major', labelsize=15)

fig.suptitle("P. fremontii leaf temperature distributions around date mean by treatment and adaptation", size = 30)
fig.supylabel('Normalized Bin Counts', size = 24)
fig.supxlabel('Leaf temperature distribution around date mean (ºC)', size = 24)

plt.tight_layout()

plt.savefig("charts/treatment_and_adaptation_normalized_hist_date_ave.png", dpi=300)

plt.show()

plt.clf()


##################################################################################################
################################################################################################## 


'''BY TREATMENT AND BLOCK --- NON-NORMALIZED HISTS ONLY -- DATEBLOCK-AVERAGED'''

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True,  figsize = (20, 10))

ax1.hist(block3_warm_dist_minus_dateblock_mean, color="green", range=(-20, 20),
          alpha=0.5, bins=100, label="Warm-adapted\npopulations")
ax1.hist(block3_cool_dist_minus_dateblock_mean, color="red", range=(-20, 20),
          alpha=0.5, bins=100, label="Cool-adapted\npopulations")
ax1.set_title('Irrigated', size = 20)
ax1.set_xlabel('Leaf temperature distribution around treatment mean (ºC)', size = 14)
ax1.legend(prop = {"size": 20})
ax1.text(-5,40000, "Number of pixels:\n{:.0f}k".format(len(block3_warm_dist_minus_dateblock_mean) / 1000), horizontalalignment = "right", size = 18)
ax1.text(10,5000, "Number of pixels:\n{:.0f}k".format(len(block3_cool_dist_minus_dateblock_mean) / 1000), horizontalalignment='left', size = 18)


ax2.hist(block2_warm_dist_minus_dateblock_mean, color="green", range=(-20, 20),
          alpha=0.5, bins=100, label="Warm-adapted\npopulations")
ax2.hist(block2_cool_dist_minus_dateblock_mean, color="red", range=(-20, 20),
          alpha=0.5, bins=100, label="Cool-adapted\npopulations")
ax2.set_xlabel('Leaf temperature distribution around treatment mean (ºC)', size = 14)
ax2.set_title('Droughted', size = 20)
ax2.legend(prop = {"size": 20})
ax2.text(-5,40000, "Number of pixels:\n{:.0f}k".format(len(block2_warm_dist_minus_dateblock_mean) / 1000), horizontalalignment = "right", size = 18)
ax2.text(10,5000, "Number of pixels:\n{:.0f}k".format(len(block2_cool_dist_minus_dateblock_mean) / 1000), horizontalalignment='left', size = 18)


fig.suptitle("P. fremontii leaf temperature distributions around mean by treatment and adaptation", size = 30)
fig.supylabel('Bin Counts', size = 24)

plt.tight_layout()

plt.savefig("charts/treatment_adaptation_absolute_value_hists.png", dpi=300)

plt.show()

plt.clf()

##################################################################################################
################################################################################################## 



# density = stats.gaussian_kde(block3_warm_dist_minus_dateblock_mean)
# n, x, _ = plt.hist(block3_warm_dist_minus_dateblock_mean, bins=np.linspace(-20, 20, 100),
#                     histtype=u'step', density=True, color="blue", label="warm2")

# density1 = stats.gaussian_kde(block2_warm_dist_minus_dateblock_mean)
# n, y, _ = plt.hist(block3_warm_dist_minus_dateblock_mean, bins=np.linspace(-20, 20, 100),
#                     histtype=u'step', density=True, color="green", label="warm3")

# density2 = stats.gaussian_kde(block3_cool_dist_minus_dateblock_mean)
# n, z, _ = plt.hist(block3_warm_dist_minus_dateblock_mean, bins=np.linspace(-20, 20, 100),
#                     histtype=u'step', density=True, color="orange", label="cool2")

# density3 = stats.gaussian_kde(block2_cool_dist_minus_dateblock_mean)
# n, a, _ = plt.hist(block3_warm_dist_minus_dateblock_mean, bins=np.linspace(-20, 20, 100),
#                     histtype=u'step', density=True, color="red", label="cool3")
# plt.plot(x, density(x), y, density1(y), z, density2(z), a, density3(a))

# plt.show()


# tcrit = 50

# cool2_bool = cool2_abs_temp > tcrit
# warm2_bool = warm2_abs_temp > tcrit
# cool3_bool = cool3_abs_temp > tcrit
# warm3_bool = warm3_abs_temp > tcrit

# # count = np.count_nonzero(arr)
# cool2_perc_above = (np.count_nonzero(cool2_bool) / len(cool2_abs_temp) * 100)
# warm2_perc_above = (np.count_nonzero(warm2_bool) / len(warm2_abs_temp) * 100)
# cool3_perc_above = (np.count_nonzero(cool3_bool) / len(cool3_abs_temp) * 100)
# warm3_perc_above = (np.count_nonzero(warm3_bool) / len(warm3_abs_temp) * 100)

# print("percentage of Block 2 cool-adapted leaves above {}ºC: {:.2f}%".format(tcrit, cool2_perc_above))
# print("percentage of Block 2 warm-adapted leaves above {}ºC: {:.2f}%".format(tcrit, warm2_perc_above))
# print("percentage of Block 3 cool-adapted leaves above {}ºC: {:.2f}%".format(tcrit, cool3_perc_above))
# print("percentage of Block 3 warm-adapted leaves above {}ºC: {:.2f}%".format(tcrit, warm3_perc_above))

# data = [cool2_perc_above, warm2_perc_above, cool3_perc_above, warm3_perc_above]
# plt.bar(x = ['cool2', 'warm2', 'cool3', 'warm3'], height = data)
# plt.show()

# cool_overall_np.mean() - warm_overall_np.mean()

# data = [overall_warm_dist, overall_cool_dist]
# labels = ["overall_warm_dist", "overall_cool_dist"]

# fig = plt.figure(figsize =(10, 7))
 
# # Creating axes instance
# ax = fig.add_axes([0, 0, 1, 1])
 
# # Creating plot
# bp = ax.boxplot(data, showfliers = False, labels = labels, whis = (10,90))
 
# # show plot
# plt.show()

###### Main #######







