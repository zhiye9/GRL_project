#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nilearn as nil
from nilearn import datasets
import numpy as np
from Install_PSM import Matcher
import pandas as pd
from nilearn.decomposition import CanICA
import nibabel as nib
from nilearn.connectome import ConnectivityMeasure



#Load data 
pcp = nil.datasets.fetch_abide_pcp(pipeline ='ccs', DSM_IV_TR = [0 ,1])
df = pd.DataFrame(data = np.c_[pcp.phenotypic.SUB_ID, pcp.func_preproc, pcp.phenotypic.AGE_AT_SCAN, pcp.phenotypic.SEX, pcp.phenotypic.DSM_IV_TR], columns= ['eid', 'file', 'age', 'sex', 'autism'])

#Propensity socre matching healthy and autism subjects by age and sex
HC_match = df[df['autism'] == '0']
DO_match = df[df['autism'] == '1']

HC_match[["age"]] = HC_match [["age"]].astype(float)
HC_match[["sex"]] = HC_match [["sex"]].astype(int)
HC_match[["autism"]] = HC_match [["autism"]].astype(int)

DO_match[["age"]] = DO_match [["age"]].astype(float)
DO_match[["sex"]] = DO_match [["sex"]].astype(int)
DO_match[["autism"]] = DO_match [["autism"]].astype(int)

HC_DO_match = pd.concat([DO_match, HC_match])
match_PSM = Matcher(DO_match, HC_match, yvar="autism", exclude=['file', 'eid'])
np.random.seed(6)
match_PSM.fit_scores(balance=True, nmodels = 6000)
match_PSM.match(method = 'min', nmatches = 2, threshold = 0.001)
HC_DO_matched = match_PSM.matched_data[['eid', 'file', 'age', 'sex', 'autism']].sort_values('autism', ascending = False)
match_PSM.predict_scores()
#np.unique(np.unique(HC_DO_matched[HC_DO_matched['autism'] == 0]['file'], return_counts = True)[1], return_counts = True)
HC_DO_matched_NoRep = HC_DO_matched.drop_duplicates(subset = ['file'])
HC_DO_matched_NoRep.reset_index(drop = True, inplace = True)

#Only keep subjects from age 21 to 33
df_age21_33 = HC_DO_matched_NoRep[(HC_DO_matched_NoRep['age'] > 21) & (HC_DO_matched_NoRep['age'] < 33)]
df_age21_33.to_csv('df_age21_33.csv', index = False)

'''
path = '/content/drive/GRL/Project/'

HC_DO_matched_NoRep.to_csv('HC_DO_matched_NoRep.csv', index = False)
!cp HC_DO_matched_NoRep.csv '/content/drive/My Drive/GRL/Project/'

HC_DO_matched_NoRep = pd.read_csv('/content/drive/My Drive/GRL/Project/HC_DO_matched_NoRep.csv')
HC_DO_matched_NoRep['eid'] == HC_DO_matched_NoRep['eid'].astype(str)
'''

#Compare distribution plots of age before and after PSM
def plot_PS_score(data, yvar, label):
  sns.distplot(data[data[yvar]==0].age, label=label[0])
  sns.distplot(data[data[yvar]==1].age, label=label[1])
  plt.legend(loc='upper right')
  #plt.xlim((0, 1))
  plt.title("Propensity Scores Before Matching")
  plt.ylabel("Percentage (%)")
  plt.xlabel("Scores")

plot_PS_score(HC_DO_matched, yvar = 'autism', label = ['Healthy Control', 'Autism'])
plot_PS_score(HC_DO_match, yvar = 'autism', label = ['Healthy Control', 'Autism'])

#Load resting-state functional MRI and compute correlation matrix
df_age21_33 = HC_DO_matched_NoRep[(HC_DO_matched_NoRep['age'] > 21) & (HC_DO_matched_NoRep['age'] < 33)]
mri_ls = df_age21_33['file']
canica = CanICA(n_components=25,
                memory="nilearn_cache", memory_level = 2, n_jobs = -1,
                mask_strategy='whole-brain-template',
                random_state=0)
                
ica = canica.fit_transform(mri_ls)                
connectome_measure = ConnectivityMeasure(kind='correlation', vectorize=True)
cor_matrix = connectome_measure.fit_transform(ica)            

#Save extracted correlation matrix

for i in range(len(cor_matrix)):
  np.savetxt('%s.txt' % (df_test.phenotypic.SUB_ID[i]), cor_matrix[i])
                
