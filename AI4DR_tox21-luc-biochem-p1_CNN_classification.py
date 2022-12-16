#!/usr/bin/env python
# coding: utf-8

import math
import os
import pandas as pd
import numpy as np
from numpy import loadtxt

import ast
from ast import literal_eval

import matplotlib as mpl
import matplotlib.pyplot as plt

from keras.models import load_model

from sklearn import metrics
from sklearn.neural_network import MLPClassifier 


import pickle
import datetime as dt

mpl.use('agg')


models_topdir = './trained_models'



curve_type_model_file = '200610_AI4DR_Shape_CNN_13classes.h5'
curve_type_model_filepath = os.path.join(models_topdir,curve_type_model_file)
curve_type_model = load_model(curve_type_model_filepath)
# summarize model.
curve_type_model.summary()

curve_type_translation_dict = {0 : "CATOP", 1 : "CANB", 2 : "CASIG", 3 : "CANT", 4 : "CAHS", 5 : "CNA", 6 : "P", 7 : "NT", 8 : "LS", 9 : "B", 10 : "B", 11 : "W", 12 : "LU"}

normalizer_model_file = '200626_AI4DR_Dispersion_Normalizer.pkl'
normalizer_model_filepath = os.path.join(models_topdir,normalizer_model_file)

dispersion_model_file = '200626_AI4DR_Dispersion_Classifier.pkl'
dispersion_model_filepath = os.path.join(models_topdir,dispersion_model_file)


# Load normalizer and dispersion models
with open(normalizer_model_filepath, "rb") as f:
    normalizer_model = pickle.load(f)

with open(dispersion_model_filepath, "rb") as f:
    dispersion_model = pickle.load(f)



def arrays_to_curve_image(pX_list, Y_list):
    pX_list = eval(pX_list.replace('nan','None'))
    Y_list = eval(Y_list.replace('nan','None'))
    assert len(pX_list) == len(Y_list) , 'Different number of replica for concentrations and inhibitions'
    mpl.rcParams["figure.dpi"] = 100
    fig, ax = plt.subplots(figsize = (1.5,1.5))
    for pX,Y in zip(pX_list,Y_list):
        assert len(pX) == len(Y) , 'Different number of concentrations and inhibitions values'
        curr_line, = ax.plot(pX,Y,'ko')
        plt.setp(curr_line, markersize=3)
    plt.xticks([])
    plt.ylim(-50.0, 150.0)
    ax.yaxis.set_tick_params(labelsize=5)
    plt.tight_layout(pad=0.5)
    img = fig.canvas.draw()

    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    W = [0.2, 0.5, 0.3]
    W_mean = np.tensordot(data, W, axes = ((-1, -1)))[..., None]
    data[:] = W_mean.astype(data.dtype)
    output = data[:,:,0]/255.0
    plt.close(fig)
    return output

def curves_to_predictions(curves):
    curves_array = curves.values 
    curves_array = np.stack(curves_array, axis=0) 
    curves_array = np.expand_dims(curves_array, axis=-1) 
    curve_type_model_output = curve_type_model.predict(curves_array)
    curve_type_model_categories = [ curve_type_translation_dict[list(probabilities).index(max(list(probabilities)))] for probabilities in curve_type_model_output]
    curve_type_model_probabilities = [ max(list(probabilities)) for probabilities in curve_type_model_output]
    return curve_type_model_categories, curve_type_model_probabilities

def sorted_positive_diff(y1,y2):
    pos_diffs = []
    for i,j in zip(y1,y2):
        if (i is None) or (j is None):
            pos_diffs.append(0)
        else:
            pos_diffs.append(abs(j-i))
    pos_diffs = sorted(pos_diffs, key=float)
    return pos_diffs

def sorted_positive_diff_triplicate(y1,y2,y3):
    pos_diffs = []
    for i,j,k in zip(y1,y2,y3):
        if ((i is None) and (j is None)) or ((i is None) and (k is None)) or ((k is None) and (j is None)):
            pos_diffs.append(0)
        else:
            if (i is None):
                 pos_diffs.append(abs(j-k))
            if (j is None):
                 pos_diffs.append(abs(i-k))
            if (k is None):
                 pos_diffs.append(abs(i-j))
            else:            
                pos_diffs.append(max([abs(j-i),abs(k-i),abs(j-k)])*2.0/3.0)
    pos_diffs = sorted(pos_diffs, key=float)
    return pos_diffs

def clean_from_nans(y):
    if 'nan' in y :
        y = eval(y.replace('nan','999.999'))
        cols_toremove = np.where(np.array(y) == 999.999)[1]
        y = np.delete(np.array(y),cols_toremove,axis=1)
        out=str(y.tolist())
    else:
        out=y
    return(out)

# ## Loading raw input file


datasets_topdir = './tox21-luc-biochem-p1'
in_file ='tox21-luc-biochem-p1.txt'
in_filepath = os.path.join(datasets_topdir,in_file)

my_df = pd.read_csv(in_filepath,sep='\t',index_col=False)


xArray_df = my_df[['CONC0', 'CONC1', 'CONC2', 'CONC3', 'CONC4', 'CONC5', 'CONC6', 'CONC7','CONC8', 'CONC9', 'CONC10', 'CONC11', 'CONC12', 'CONC13', 'CONC14']]
xArray_half_df = my_df[['CONC0', 'CONC2', 'CONC4', 'CONC6', 'CONC8', 'CONC10', 'CONC12', 'CONC14']]

Y_df = my_df[['DATA0','DATA1', 'DATA2', 'DATA3', 'DATA4', 'DATA5', 'DATA6', 'DATA7', 'DATA8','DATA9', 'DATA10', 'DATA11', 'DATA12', 'DATA13', 'DATA14']]
Y_half_df = my_df[['DATA0', 'DATA2', 'DATA4',  'DATA6', 'DATA8', 'DATA10', 'DATA12', 'DATA14']]

my_df['pX'] = [list(-np.log10(xArray_df.iloc[i])) for i in xArray_df.index]
my_df['pXhalf'] = [list(-np.log10(xArray_half_df.iloc[i])) for i in xArray_half_df.index]

my_df['Y'] = [list(Y_df.iloc[i]) for i in Y_df.index]
my_df['Yhalf'] = [list(Y_half_df.iloc[i]) for i in Y_half_df.index]


work_df = my_df[['SAMPLE_ID', 'SAMPLE_DATA_TYPE','pX', 'pXhalf', 'Y', 'Yhalf']]

# We will only use the biochemical assay data, not the viability one
work_df = work_df[[not s.startswith('viability') for s in work_df['SAMPLE_DATA_TYPE']]]


# Build DRC data for the different cases we consider :
# Starting from DRC with 3 replica at 15 concentrations 
# We generate DRCs with two of the three replica
# Either at 15 concentrations or at 8 concentrations
# In each case, the raw data needs to be translated 
# See Huang, R. et al. Modelling the Tox21 10 K chemical profiles for in vivo toxicity prediction and mechanism characterization. Nat Commun 7, 10425 (2016).
  

curves_df = pd.DataFrame()
curr_curve_dict = {}
for i,g in work_df.sort_values(['SAMPLE_ID', 'SAMPLE_DATA_TYPE'],ascending=True).groupby('SAMPLE_ID'):
    #print(i, g.shape[0])
    if g.shape[0] == 3: 
        curr_curve_dict['SAMPLE_ID'] = i
        curr_curve_dict['pX01_list'] = str(list(g['pX'][0:2]))
        curr_curve_dict['pX02_list'] = str(list(g['pX'][0:3:2]))
        curr_curve_dict['pX12_list'] = str(list(g['pX'][1:3]))
        curr_curve_dict['pX_list'] = str(list(g['pX']))
        
        curr_curve_dict['pXhalf01_list'] = str(list(g['pXhalf'][0:2]))
        curr_curve_dict['pXhalf02_list'] = str(list(g['pXhalf'][0:3:2]))
        curr_curve_dict['pXhalf12_list'] = str(list(g['pXhalf'][1:3]))
        curr_curve_dict['pXhalf_list'] = str(list(g['pXhalf']))

        Y_translation01 = np.nanmean([np.nanmin(g['Y'].iloc[0]),np.nanmin(g['Y'].iloc[1])])
        Y_translation02 = np.nanmean([np.nanmin(g['Y'].iloc[0]),np.nanmin(g['Y'].iloc[2])])
        Y_translation12 = np.nanmean([np.nanmin(g['Y'].iloc[1]),np.nanmin(g['Y'].iloc[2])])
        Y_translation123 = np.nanmean([np.nanmin(g['Y'].iloc[0]),np.nanmin(g['Y'].iloc[1]),np.nanmin(g['Y'].iloc[2])])
        
        curr_curve_dict['Y01_list'] = str(list([list(np.array(z)-Y_translation01) for z in g['Y'][0:2]]))
        curr_curve_dict['Y02_list'] = str(list([list(np.array(z)-Y_translation02) for z in g['Y'][0:3:2]]))
        curr_curve_dict['Y12_list'] = str(list([list(np.array(z)-Y_translation12) for z in g['Y'][1:3]]))
        curr_curve_dict['Y_list'] = str(list([list(np.array(z)-Y_translation123) for z in g['Y']]))
        curr_curve_dict['Y01_list_notr'] = str(list([list(np.array(z)) for z in g['Y'][0:2]]))
        curr_curve_dict['Y02_list_notr'] = str(list([list(np.array(z)) for z in g['Y'][0:3:2]]))
        curr_curve_dict['Y12_list_notr'] = str(list([list(np.array(z)) for z in g['Y'][1:3]]))
        curr_curve_dict['Y_list_notr'] = str(list([list(np.array(z)) for z in g['Y']]))
                                       
        Y_translationhalf01 = np.nanmean([np.nanmin(g['Yhalf'].iloc[0]),np.nanmin(g['Yhalf'].iloc[1])])
        Y_translationhalf02 = np.nanmean([np.nanmin(g['Yhalf'].iloc[0]),np.nanmin(g['Yhalf'].iloc[2])])
        Y_translationhalf12 = np.nanmean([np.nanmin(g['Yhalf'].iloc[1]),np.nanmin(g['Yhalf'].iloc[2])])
        Y_translationhalf123 = np.nanmean([np.nanmin(g['Yhalf'].iloc[0]),np.nanmin(g['Yhalf'].iloc[1]),np.nanmin(g['Yhalf'].iloc[2])])
        
        curr_curve_dict['Yhalf01_list'] = str(list([list(np.array(z)-Y_translationhalf01) for z in g['Yhalf'][0:2]]))
        curr_curve_dict['Yhalf02_list'] = str(list([list(np.array(z)-Y_translationhalf02) for z in g['Yhalf'][0:3:2]]))
        curr_curve_dict['Yhalf12_list'] = str(list([list(np.array(z)-Y_translationhalf12) for z in g['Yhalf'][1:3]]))
        curr_curve_dict['Yhalf_list'] = str(list([list(np.array(z)-Y_translationhalf123) for z in g['Yhalf']]))
        curr_curve_dict['Yhalf01_list_notr'] = str(list([list(np.array(z)) for z in g['Yhalf'][0:2]]))
        curr_curve_dict['Yhalf02_list_notr'] = str(list([list(np.array(z)) for z in g['Yhalf'][0:3:2]]))
        curr_curve_dict['Yhalf12_list_notr'] = str(list([list(np.array(z)) for z in g['Yhalf'][1:3]]))
        curr_curve_dict['Yhalf_list_notr'] = str(list([list(np.array(z)) for z in g['Yhalf']]))
        curves_df = curves_df.append(pd.DataFrame(curr_curve_dict, index=[0]))
    else:
        pass

# Data needed for the dispersion classifier

curves_df['Y0Y1Y2diff'] = [sorted_positive_diff_triplicate(eval(y.replace('nan','0'))[0],eval(y.replace('nan','0'))[1],eval(y.replace('nan','0'))[2]) for y in curves_df['Y_list']]
curves_df['Y0Y1diff'] = [sorted_positive_diff(eval(clean_from_nans(y))[0],eval(clean_from_nans(y))[1]) for y in curves_df['Y01_list']]
curves_df['Y0Y2diff'] = [sorted_positive_diff(eval(clean_from_nans(y))[0],eval(clean_from_nans(y))[1]) for y in curves_df['Y02_list']]
curves_df['Y1Y2diff'] = [sorted_positive_diff(eval(clean_from_nans(y))[0],eval(clean_from_nans(y))[1]) for y in curves_df['Y12_list']]

curves_df['Y0Y1Y2halfdiff'] = [sorted_positive_diff_triplicate(eval(y.replace('nan','0'))[0],eval(y.replace('nan','0'))[1],eval(y.replace('nan','0'))[2]) for y in curves_df['Yhalf_list']]
curves_df['Y0Y1halfdiff'] = [sorted_positive_diff(eval(clean_from_nans(y))[0],eval(clean_from_nans(y))[1]) for y in curves_df['Yhalf01_list']]
curves_df['Y0Y2halfdiff'] = [sorted_positive_diff(eval(clean_from_nans(y))[0],eval(clean_from_nans(y))[1]) for y in curves_df['Yhalf02_list']]
curves_df['Y1Y2halfdiff'] = [sorted_positive_diff(eval(clean_from_nans(y))[0],eval(clean_from_nans(y))[1]) for y in curves_df['Yhalf12_list']]

curves_df['Y0Y1Y2diff_q1'] = [np.percentile(d, 25) for d in curves_df['Y0Y1Y2diff'] ]
curves_df['Y0Y1Y2diff_q2'] = [np.percentile(d, 50) for d in curves_df['Y0Y1Y2diff'] ]
curves_df['Y0Y1Y2diff_q3'] = [np.percentile(d, 75) for d in curves_df['Y0Y1Y2diff']]
curves_df['Y0Y1Y2diff_interquartile'] = [q3-q1 for q1,q3 in zip(curves_df['Y0Y1Y2diff_q1'],curves_df['Y0Y1Y2diff_q3'])]

curves_df['Y0Y1diff_q1'] = [np.percentile(d, 25) for d in curves_df['Y0Y1diff'] ]
curves_df['Y0Y1diff_q2'] = [np.percentile(d, 50) for d in curves_df['Y0Y1diff'] ]
curves_df['Y0Y1diff_q3'] = [np.percentile(d, 75) for d in curves_df['Y0Y1diff']]
curves_df['Y0Y1diff_interquartile'] = [q3-q1 for q1,q3 in zip(curves_df['Y0Y1diff_q1'],curves_df['Y0Y1diff_q3'])]

curves_df['Y0Y2diff_q1'] = [np.percentile(d, 25) for d in curves_df['Y0Y2diff'] ]
curves_df['Y0Y2diff_q2'] = [np.percentile(d, 50) for d in curves_df['Y0Y2diff'] ]
curves_df['Y0Y2diff_q3'] = [np.percentile(d, 75) for d in curves_df['Y0Y2diff']]
curves_df['Y0Y2diff_interquartile'] = [q3-q1 for q1,q3 in zip(curves_df['Y0Y2diff_q1'],curves_df['Y0Y2diff_q3'])]

curves_df['Y1Y2diff_q1'] = [np.percentile(d, 25) for d in curves_df['Y1Y2diff'] ]
curves_df['Y1Y2diff_q2'] = [np.percentile(d, 50) for d in curves_df['Y1Y2diff'] ]
curves_df['Y1Y2diff_q3'] = [np.percentile(d, 75) for d in curves_df['Y1Y2diff']]
curves_df['Y1Y2diff_interquartile'] = [q3-q1 for q1,q3 in zip(curves_df['Y1Y2diff_q1'],curves_df['Y1Y2diff_q3'])]


curves_df['Y0Y1Y2halfdiff_q1'] = [np.percentile(d, 25) for d in curves_df['Y0Y1Y2halfdiff'] ]
curves_df['Y0Y1Y2halfdiff_q2'] = [np.percentile(d, 50) for d in curves_df['Y0Y1Y2halfdiff'] ]
curves_df['Y0Y1Y2halfdiff_q3'] = [np.percentile(d, 75) for d in curves_df['Y0Y1Y2halfdiff']]
curves_df['Y0Y1Y2halfdiff_interquartile'] = [q3-q1 for q1,q3 in zip(curves_df['Y0Y1Y2halfdiff_q1'],curves_df['Y0Y1Y2halfdiff_q3'])]

curves_df['Y0Y1halfdiff_q1'] = [np.percentile(d, 25) for d in curves_df['Y0Y1halfdiff'] ]
curves_df['Y0Y1halfdiff_q2'] = [np.percentile(d, 50) for d in curves_df['Y0Y1halfdiff'] ]
curves_df['Y0Y1halfdiff_q3'] = [np.percentile(d, 75) for d in curves_df['Y0Y1halfdiff']]
curves_df['Y0Y1halfdiff_interquartile'] = [q3-q1 for q1,q3 in zip(curves_df['Y0Y1halfdiff_q1'],curves_df['Y0Y1halfdiff_q3'])]

curves_df['Y0Y2halfdiff_q1'] = [np.percentile(d, 25) for d in curves_df['Y0Y2halfdiff'] ]
curves_df['Y0Y2halfdiff_q2'] = [np.percentile(d, 50) for d in curves_df['Y0Y2halfdiff'] ]
curves_df['Y0Y2halfdiff_q3'] = [np.percentile(d, 75) for d in curves_df['Y0Y2halfdiff']]
curves_df['Y0Y2halfdiff_interquartile'] = [q3-q1 for q1,q3 in zip(curves_df['Y0Y2halfdiff_q1'],curves_df['Y0Y2halfdiff_q3'])]

curves_df['Y1Y2halfdiff_q1'] = [np.percentile(d, 25) for d in curves_df['Y1Y2halfdiff'] ]
curves_df['Y1Y2halfdiff_q2'] = [np.percentile(d, 50) for d in curves_df['Y1Y2halfdiff'] ]
curves_df['Y1Y2halfdiff_q3'] = [np.percentile(d, 75) for d in curves_df['Y1Y2halfdiff']]
curves_df['Y1Y2halfdiff_interquartile'] = [q3-q1 for q1,q3 in zip(curves_df['Y1Y2halfdiff_q1'],curves_df['Y1Y2halfdiff_q3'])]


# Dispersion classification
# Step 1 : normalization

norm_diff_012_df = pd.DataFrame(normalizer_model.transform(curves_df[['Y0Y1Y2diff_q1','Y0Y1Y2diff_q2','Y0Y1Y2diff_q3','Y0Y1Y2diff_interquartile']]),columns=['Y0Y1Y2diff_q1_norm','Y0Y1Y2diff_q2_norm','Y0Y1Y2diff_q3_norm','Y0Y1Y2diff_interquartile_norm'])
norm_diff_01_df = pd.DataFrame(normalizer_model.transform(curves_df[['Y0Y1diff_q1','Y0Y1diff_q2','Y0Y1diff_q3','Y0Y1diff_interquartile']]),columns=['Y0Y1diff_q1_norm','Y0Y1diff_q2_norm','Y0Y1diff_q3_norm','Y0Y1diff_interquartile_norm'])
norm_diff_02_df = pd.DataFrame(normalizer_model.transform(curves_df[['Y0Y2diff_q1','Y0Y2diff_q2','Y0Y2diff_q3','Y0Y2diff_interquartile']]),columns=['Y0Y2diff_q1_norm','Y0Y2diff_q2_norm','Y0Y2diff_q3_norm','Y0Y2diff_interquartile_norm'])
norm_diff_12_df = pd.DataFrame(normalizer_model.transform(curves_df[['Y1Y2diff_q1','Y1Y2diff_q2','Y1Y2diff_q3','Y1Y2diff_interquartile']]),columns=['Y1Y2diff_q1_norm','Y1Y2diff_q2_norm','Y1Y2diff_q3_norm','Y1Y2diff_interquartile_norm'])

norm_halfdiff_012_df = pd.DataFrame(normalizer_model.transform(curves_df[['Y0Y1Y2halfdiff_q1','Y0Y1Y2halfdiff_q2','Y0Y1Y2halfdiff_q3','Y0Y1Y2halfdiff_interquartile']]),columns=['Y0Y1Y2halfdiff_q1_norm','Y0Y1Y2halfdiff_q2_norm','Y0Y1Y2halfdiff_q3_norm','Y0Y1Y2halfdiff_interquartile_norm'])
norm_halfdiff_01_df = pd.DataFrame(normalizer_model.transform(curves_df[['Y0Y1halfdiff_q1','Y0Y1halfdiff_q2','Y0Y1halfdiff_q3','Y0Y1halfdiff_interquartile']]),columns=['Y0Y1halfdiff_q1_norm','Y0Y1halfdiff_q2_norm','Y0Y1halfdiff_q3_norm','Y0Y1halfdiff_interquartile_norm'])
norm_halfdiff_02_df = pd.DataFrame(normalizer_model.transform(curves_df[['Y0Y2halfdiff_q1','Y0Y2halfdiff_q2','Y0Y2halfdiff_q3','Y0Y2halfdiff_interquartile']]),columns=['Y0Y2halfdiff_q1_norm','Y0Y2halfdiff_q2_norm','Y0Y2halfdiff_q3_norm','Y0Y2halfdiff_interquartile_norm'])
norm_halfdiff_12_df = pd.DataFrame(normalizer_model.transform(curves_df[['Y1Y2halfdiff_q1','Y1Y2halfdiff_q2','Y1Y2halfdiff_q3','Y1Y2halfdiff_interquartile']]),columns=['Y1Y2halfdiff_q1_norm','Y1Y2halfdiff_q2_norm','Y1Y2halfdiff_q3_norm','Y1Y2halfdiff_interquartile_norm'])

# Dispersion classification
# Step 1 : classification

curves_df['Disp_model012'] = dispersion_model.predict(norm_diff_012_df)
curves_df['Disp_model01'] = dispersion_model.predict(norm_diff_01_df)
curves_df['Disp_model02'] = dispersion_model.predict(norm_diff_02_df)
curves_df['Disp_model12'] = dispersion_model.predict(norm_diff_12_df)

curves_df['Disp_modelhalf012'] = dispersion_model.predict(norm_halfdiff_012_df)
curves_df['Disp_modelhalf01'] = dispersion_model.predict(norm_halfdiff_01_df)
curves_df['Disp_modelhalf02'] = dispersion_model.predict(norm_halfdiff_02_df)
curves_df['Disp_modelhalf12'] = dispersion_model.predict(norm_halfdiff_12_df)

disp_model_probs012 = pd.DataFrame(dispersion_model.predict_proba(norm_diff_012_df),columns=['Disp_Proba012','NoDisp_Proba012'])
disp_model_probs01 = pd.DataFrame(dispersion_model.predict_proba(norm_diff_01_df),columns=['Disp_Proba01','NoDisp_Proba01'])
disp_model_probs02 = pd.DataFrame(dispersion_model.predict_proba(norm_diff_02_df),columns=['Disp_Proba02','NoDisp_Proba02'])
disp_model_probs12 = pd.DataFrame(dispersion_model.predict_proba(norm_diff_12_df),columns=['Disp_Proba12','NoDisp_Proba12'])

disp_model_probshalf012 = pd.DataFrame(dispersion_model.predict_proba(norm_halfdiff_012_df),columns=['Disp_Probahalf012','NoDisp_Probahalf012'])
disp_model_probshalf01 = pd.DataFrame(dispersion_model.predict_proba(norm_halfdiff_01_df),columns=['Disp_Probahalf01','NoDisp_Probahalf01'])
disp_model_probshalf02 = pd.DataFrame(dispersion_model.predict_proba(norm_halfdiff_02_df),columns=['Disp_Probahalf02','NoDisp_Probahalf02'])
disp_model_probshalf12 = pd.DataFrame(dispersion_model.predict_proba(norm_halfdiff_12_df),columns=['Disp_Probahalf12','NoDisp_Probahalf12'])


curves_df = pd.concat([curves_df.reset_index(drop=True),disp_model_probs012,disp_model_probs01,disp_model_probs02,disp_model_probs12,disp_model_probshalf012,disp_model_probshalf01,disp_model_probshalf02,disp_model_probshalf12],axis=1)


# Shape classification
# Step 1 : DRC image generation

all_curves = []
all_curves01 = []
all_curves02 = []
all_curves12 = []
all_curveshalf = []
all_curveshalf01 = []
all_curveshalf02 = []
all_curveshalf12 = []
                                       
for i in range(0, len(curves_df), 1000):
    print(i)
    slc = curves_df.iloc[i : i + 1000]
    all_curves01.append([arrays_to_curve_image(x,y) for x,y in zip(slc['pX01_list'],slc['Y01_list'])])
    all_curves02.append([arrays_to_curve_image(x,y) for x,y in zip(slc['pX02_list'],slc['Y02_list'])])
    all_curves12.append([arrays_to_curve_image(x,y) for x,y in zip(slc['pX12_list'],slc['Y12_list'])])        
    all_curves.append([arrays_to_curve_image(x,y) for x,y in zip(slc['pX_list'],slc['Y_list'])])

    all_curveshalf01.append([arrays_to_curve_image(x,y) for x,y in zip(slc['pXhalf01_list'],slc['Yhalf01_list'])])
    all_curveshalf02.append([arrays_to_curve_image(x,y) for x,y in zip(slc['pXhalf02_list'],slc['Yhalf02_list'])])
    all_curveshalf12.append([arrays_to_curve_image(x,y) for x,y in zip(slc['pXhalf12_list'],slc['Yhalf12_list'])])        
    all_curveshalf.append([arrays_to_curve_image(x,y) for x,y in zip(slc['pXhalf_list'],slc['Yhalf_list'])])

curves_df['curves'] = [item for subset in all_curves for item in subset ]
curves_df['curves01'] = [item for subset in all_curves01 for item in subset ]
curves_df['curves02'] = [item for subset in all_curves02 for item in subset ]
curves_df['curves12'] = [item for subset in all_curves12 for item in subset ]

curves_df['curveshalf'] = [item for subset in all_curves for item in subset ]
curves_df['curveshalf01'] = [item for subset in all_curveshalf01 for item in subset ]
curves_df['curveshalf02'] = [item for subset in all_curveshalf02 for item in subset ]
curves_df['curveshalf12'] = [item for subset in all_curveshalf12 for item in subset ]

# Shape classification
# Step 2 : DRC shape classification

curves_df['category'], curves_df['probability'] = curves_to_predictions(curves_df['curves'])
curves_df['category01'], curves_df['probability01'] = curves_to_predictions(curves_df['curves01'])
curves_df['category02'], curves_df['probability02'] = curves_to_predictions(curves_df['curves02'])
curves_df['category12'], curves_df['probability12'] = curves_to_predictions(curves_df['curves12'])

curves_df['categoryhalf'], curves_df['probabilityhalf'] = curves_to_predictions(curves_df['curveshalf'])
curves_df['categoryhalf01'], curves_df['probabilityhalf01'] = curves_to_predictions(curves_df['curveshalf01'])
curves_df['categoryhalf02'], curves_df['probabilityhalf02'] = curves_to_predictions(curves_df['curveshalf02'])
curves_df['categoryhalf12'], curves_df['probabilityhalf12'] = curves_to_predictions(curves_df['curveshalf12'])


curves_df.drop(columns=['Y0Y1Y2diff','Y0Y1diff', 'Y0Y2diff', 'Y1Y2diff', 
                        'Y0Y1Y2diff_q1', 'Y0Y1Y2diff_q2', 'Y0Y1Y2diff_q3', 'Y0Y1Y2diff_interquartile', 
                        'Y0Y1diff_q1', 'Y0Y1diff_q2', 'Y0Y1diff_q3', 'Y0Y1diff_interquartile', 
                        'Y0Y2diff_q1', 'Y0Y2diff_q2', 'Y0Y2diff_q3', 'Y0Y2diff_interquartile', 
                        'Y1Y2diff_q1', 'Y1Y2diff_q2', 'Y1Y2diff_q3', 'Y1Y2diff_interquartile', 
                        'curves', 'curves01', 'curves02', 'curves12',
                        'Y0Y1Y2halfdiff','Y0Y1halfdiff', 'Y0Y2halfdiff', 'Y1Y2halfdiff', 
                        'Y0Y1Y2halfdiff_q1', 'Y0Y1Y2halfdiff_q2', 'Y0Y1Y2halfdiff_q3', 'Y0Y1Y2halfdiff_interquartile', 
                        'Y0Y1halfdiff_q1', 'Y0Y1halfdiff_q2', 'Y0Y1halfdiff_q3', 'Y0Y1halfdiff_interquartile', 
                        'Y0Y2halfdiff_q1', 'Y0Y2diff_q2', 'Y0Y2halfdiff_q3', 'Y0Y2halfdiff_interquartile', 
                        'Y1Y2halfdiff_q1', 'Y1Y2diff_q2', 'Y1Y2halfdiff_q3', 'Y1Y2halfdiff_interquartile', 
                        'curveshalf', 'curveshalf01', 'curveshalf02', 'curveshalf12',
                       ], inplace=True)

curves_df = curves_df.merge(my_df[['SAMPLE_ID','ASSAY_OUTCOME','CURVE_CLASS2']].groupby('SAMPLE_ID').first(),on='SAMPLE_ID')

out_file ='AI4DR_annotated_tox21_luc_biochem_p1_dup.pkl'

curves_df.to_pickle(out_file)
