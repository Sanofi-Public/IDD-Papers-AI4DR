#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import math
import shutil
import pandas as pd
import numpy as np
from numpy import loadtxt
import glob
import sys
import os

import ast
from ast import literal_eval

import matplotlib as mpl
import matplotlib.pyplot as plt

from keras.models import load_model

from sklearn import metrics
from sklearn.neural_network import MLPClassifier 

from operator import add
from hillfit import fitting

import pickle
import datetime as dt

mpl.use('agg')


models_topdir = './trained_models'


# In[12]:


def extract_hill_parameters(pX_list, Y_list):
    r_2_list = []
    top_list = []
    bottom_list = []
    ec50_list = []
    nh_list = []
    for px_l,y_l in zip(pX_list,Y_list):
        px = eval(px_l)
        y = eval(y_l)
        n_replica = np.array(y).shape[0]
        mean_y = list(np.array(y).sum(axis=0)/n_replica)
        mean_y.reverse()
        x = [ 10**(-z) for z in px[0] ]
        x = list(x)
        hf = fitting.HillFit(x, mean_y)
        try:
            hf.fitting(log_x = True, generate_figure = False)
            r_2_list.append(hf.r_2)
            top_list.append(hf.top)
            bottom_list.append(hf.bottom)
            ec50_list.append(hf.ec50)
            nh_list.append(hf.nH)
        except:
            r_2_list.append(np.nan)
            top_list.append(np.nan)
            bottom_list.append(np.nan)
            ec50_list.append(np.nan)
            nh_list.append(np.nan)
        dirs = glob.glob(f'{curr_dir}/Hillfit-reg*/')
        for d in dirs:
            shutil.rmtree(d)
    return(np.array([r_2_list, top_list, bottom_list, ec50_list, nh_list]).T)


# In[59]:


def parameters_to_predictions(hill_array):
    all_categories = []
    all_probabilities = []
    all_categories_nonans = []
    all_probabilities_nonans = []
    # Manage cases where the parameter extraction did not work
    finite_indices = []
    i = 0
    for l in hill_array:
        if not True in np.isnan(l):
            finite_indices.append(i)
        i = i + 1
    hill_array_nonans = hill_array[finite_indices]
    # Perform prediction on DRC with Hill's equation parameters
    hill_array_nonans = normalizer_model.transform(hill_array_nonans)
    pred_proba_nonans = curve_type_model.predict_proba(hill_array_nonans)
    pred = curve_type_model.predict(hill_array_nonans)
    pred_proba_nonans = np.array(pred_proba_nonans).T
    for assign, probs in zip(pred,list(pred_proba_nonans[1])):
        assign_index = list(assign).index(max(list(assign)))
        proba = probs[assign_index]
        category = curve_type_translation_dict[assign_index]
        all_categories_nonans.append(category)
        all_probabilities_nonans.append(proba)
    # The category as 'XX' if Hill's parameters are missing 
    for j in range(hill_array.shape[0]):
        if j in finite_indices:
            all_categories.append(all_categories_nonans.pop(0))
            all_probabilities.append(all_probabilities_nonans.pop(0))
        else:
            all_categories.append('XX')
            all_probabilities.append(1)        
    return(all_categories,all_probabilities)


# In[4]:


curve_type_translation_dict = {0 : "CATOP", 1 : "CANB", 2 : "CASIG", 3 : "CANT", 4 : "CAHS", 5 : "CNA", 6 : "P", 7 : "NT", 8 : "LS", 9 : "B", 10 : "B", 11 : "W", 12 : "LU"}

normalizer_model_file = '221110_AI4DR_Shape_Scaler.pkl'
normalizer_model_filepath = os.path.join(models_topdir,normalizer_model_file)

curve_type_model_file = '221110_AI4DR_Shape_RF_13classes.pkl'
curve_type_model_filepath = os.path.join(models_topdir,curve_type_model_file)


# Load normalizer and dispersion models
with open(normalizer_model_filepath, "rb") as f:
    normalizer_model = pickle.load(f)

with open(curve_type_model_filepath, "rb") as f:
    curve_type_model = pickle.load(f)


# In[5]:


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


# In[6]:


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


# In[7]:


curves_df.columns


# In[8]:


curr_dir = '.'


# In[13]:


hill01_params = extract_hill_parameters(curves_df['pX01_list'],curves_df['Y01_list'])


# In[18]:


np.isnan(hill01_params.T[0])


# In[22]:


hill01_params[~np.isnan(hill01_params)]


# In[34]:


finite_indices = []
i = 0
for l in hill01_params:
    if not True in np.isnan(l):
        finite_indices.append(i)
    i = i + 1


# In[39]:


hill01_params_nonans = hill01_params[finite_indices]


# In[60]:


hill01_categories,hill01_probabilities = parameters_to_predictions(hill01_params)


# In[ ]:


curves_df['category01'], curves_df['probability01'] = parameters_to_predictions(hill01_params)


# In[64]:


hill02_params = extract_hill_parameters(curves_df['pX02_list'],curves_df['Y02_list'])
curves_df['category02'], curves_df['probability02'] = parameters_to_predictions(hill02_params)


# In[67]:


hill_params = extract_hill_parameters(curves_df['pX_list'],curves_df['Y_list'])
curves_df['category'], curves_df['probability'] = parameters_to_predictions(hill_params)
hill_half_params = extract_hill_parameters(curves_df['pXhalf_list'],curves_df['Yhalf_list'])
curves_df['categoryhalf'], curves_df['probabilityhalf'] = parameters_to_predictions(hill_half_params)
hill_half01_params = extract_hill_parameters(curves_df['pXhalf01_list'],curves_df['Yhalf01_list'])
curves_df['categoryhalf01'], curves_df['probabilityhalf01'] = parameters_to_predictions(hill_half01_params)
hill_half02_params = extract_hill_parameters(curves_df['pXhalf02_list'],curves_df['Yhalf02_list'])
curves_df['categoryhalf02'], curves_df['probabilityhalf02'] = parameters_to_predictions(hill_half02_params)
hill_half12_params = extract_hill_parameters(curves_df['pXhalf12_list'],curves_df['Yhalf12_list'])
curves_df['categoryhalf12'], curves_df['probabilityhalf12'] = parameters_to_predictions(hill_half12_params)


# In[68]:


curves_df = curves_df.merge(my_df[['SAMPLE_ID','ASSAY_OUTCOME','CURVE_CLASS2']].groupby('SAMPLE_ID').first(),on='SAMPLE_ID')


# In[69]:


out_file ='AI4DR_annotated_tox21_luc_biochem_p1_RF.pkl'

curves_df.to_pickle(out_file)

