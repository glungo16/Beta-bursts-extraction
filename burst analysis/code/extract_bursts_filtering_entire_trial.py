""" !pip install hmmlearn
!pip install mne

from hmmlearn import hmm """


import numpy as np
from sklearn.decomposition import PCA
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy import signal
from numpy.random import default_rng
from itertools import groupby
import pandas as pd
import os
from os.path import isfile, join
from typing import List


##### Important name conventions of input files:
# participant (e.g. AD0109), task (rest vs motor), session (20Hz vs 70Hz vs sham), block (base, post15, post45)
# channel (8 channels), trials (1 vs variable for motor), 
# mov interval (rest vs pre, mrbd, pmbr), bursts (variable on the number of them), burst characteristics (rate, ampl, duration)


# filename = participant_task_session_block.csv
# e.g. AD0109_motor_20Hz_post15.csv
# e.g. AD0109_rest_sham_base.csv


## TO CHANGE BASED ON DATA/COMPUTER
import_folder = r"C:\Cristi\Grad McGill\Project\Beta bursts\data_raw\stroke"
export_folder = r"C:\Cristi\Grad McGill\Project\Beta bursts\data_extracted\stroke"

# Example of epoch separation
# movement epochs [-1,8] -> pre: [-1,0]; prep: [0, 0.5]; mov: [0.5, 3.5]; rel: [3.5, 4.5]; post: [4.5, 8]
# epochs = [1, 0.5, 3, 1, 3.5]
# epochs_name = ['pre', 'prep', 'mov', 'rel', 'post']

# total length should be equal to the length of each motor epoch in Brainstorm
# [-1.5, 10.5]
# [-1.1,-0.1] - pre-movement; [0.5, 3.5] - movement; [5, 8] - post-movement
mov_epoch_intervals = [0.4, 1, 0.6, 3, 1.5, 3, 2.504]
mov_epoch_name_intervals = ['p-pre', 'pre', 'prep', 'mov', 'rel', 'post', 'p-post']



# total length should be equal to the length of each rest epoch in Brainstorm
rest_intervals = [0.3, 3, 0.3] # duration of each segment in seconds
rest_name_intervals = ['pre-int', 'int', 'post-int']


# file with headers = true; no headers = false
containsHeaders = True
# only if containsHeaders = False, manually write down the channels in the import file
# e.g. channels = ['FC1', 'FC3', 'FC5']
channels = []


betaBandPass = [15, 29]

thresholdPercentile = 75


sampleRate = 250
#HMM_state_prob = 0.66

min_burst_duration = sampleRate / 10    ## 100 ms
burst_duration_in_ms_conversion = 1000/sampleRate

# allows to completely calculate bursts that start or end within the subepoch
# make True if intervals of interest have different sizes
flexible_interval = True
flexible_interval_size_sec = 0.3


rest_state = True ### if true, automatic based on file name (motor vs rest)

########################################## HMM implementation commented out (unusable)
useHMM = False # true = HMM method; false = threshold method
############ END OF PARAMETERS


epochs = []
epochs_name = []

importFiles = [join(import_folder, f) for f in os.listdir(import_folder) if isfile(join(import_folder, f))]
 
for importFile in importFiles:

  # Parameters' initialization

  #segmentation
  imp = importFile.split("\\")[-1]
  exportFile = "bursts_" + imp
  exportFile = join(export_folder, exportFile)

  df = pd.read_csv (importFile,header=None, engine='python')
  df=df.loc[~(df==0).all(axis=1)]


  if(imp.split("_")[1] == "motor"):
    rest_state = False
  elif(imp.split("_")[1] == "rest"):
    rest_state = True
  else:
    print("Wrong input files")
    break


  if(containsHeaders):
    channels = df.iloc[1:,0]
    df = df.iloc[1:,1:]

  if(pd.isnull(df.iloc[-1,-1])):
    df = df.iloc[:,:-1]


  epochLength = 0

  if rest_state:
    epochs = rest_intervals
    epochs_name = rest_name_intervals
  else:
    epochs = mov_epoch_intervals
    epochs_name = mov_epoch_name_intervals 

  if flexible_interval:
    flexible_interval_size = int(flexible_interval_size_sec * sampleRate)
  else:
    flexible_interval_size = 0

  epochs = [int(e*sampleRate) for e in epochs]  

  epochLength = sum(epochs)








  #lo = number of segments
  #res1 = final output
  #tt = iteration of each segment/epoch
  #tt1 = index at which the current segment starts
  #tt2 = index at which the current segment ends
  #sub_epoch = the current subepoch number (e.g. pre, mov, post)
  #tt1_1 = index at which the current subepoch starts
  #tt1_2 = index at which the current subepoch ends
  #df = all data
  #df1 = data for current subepoch
  #ch = current channel number
  #b_state = z = 0 and 1 array corresponding to bursts vs non-bursts for current channel and subepoch
  #q, w = b_state probabilities for HMM

  lo = int((df.shape[1])/epochLength)

  res1 = [['']]

  res1[0].append('rate')
  res1[0].append('ampl')
  res1[0].append('duration')


  for ch in range (0,df.shape[0]):                     ## loop for each channel
  
    if rest_state:
      sos = signal.butter(5, betaBandPass, 'bandpass', fs=sampleRate, output='sos')
      filtered = signal.sosfilt(sos,df.iloc[ch, :])
      analytic_signal = signal.hilbert(filtered)
      amplitude_envelope_trial = np.abs(analytic_signal)
      amplitude_envelope_trial = signal.detrend(amplitude_envelope_trial, type = 'linear') 





    
    for tt in range (1,lo+1) :
      
      tt1 = tt*epochLength-epochLength                          ## 530 means the time point of each epoch
      tt2 = tt1 + epochLength

      tt1_1 = tt1
      tt1_2 = tt1_1


      ## Extracting beta wave from raw EEG/MEG signal on the whole trial
      if not rest_state:
        sos = signal.butter(5, betaBandPass, 'bandpass', fs=sampleRate, output='sos')
        filtered = signal.sosfilt(sos,df.iloc[ch, tt1:tt2])
        analytic_signal = signal.hilbert(filtered)
        amplitude_envelope_trial = np.abs(analytic_signal)
        amplitude_envelope_trial = signal.detrend(amplitude_envelope_trial, type = 'linear')
      
      for sub_epoch in range(len(epochs)):

        tt1_1 = tt1_2
        tt1_2 = tt1_1 + epochs[sub_epoch]

        tt1_1_flex = tt1_1 - flexible_interval_size
        tt1_2_flex = tt1_2 + flexible_interval_size

        if(tt1_1_flex < tt1):
          tt1_1_flex = tt1_1

        if(tt1_2_flex > tt2):
          tt1_2_flex = tt1_2

        tt1_1_flex_dif = tt1_1 - tt1_1_flex
        tt1_2_flex_dif = tt1_2_flex - tt1_2

        df1 = df.iloc[ch,tt1_1_flex:tt1_2_flex]                          ## current epoch
        interval = (tt1_2 - tt1_1)/sampleRate           ## duration of epoch in seconds


        

      

        retry = 0                                       ## retry to compute HMM up to 3 times in case it fails to identify bursts

        if(useHMM == False):
          retry = 2

        while(retry < 3):
          z = []
          q = []
          w = []
          b_state = []

          if rest_state:
            amplitude_envelope = amplitude_envelope_trial[tt1_1_flex:tt1_2_flex]
          else:
            amplitude_envelope = amplitude_envelope_trial[tt1_1_flex-tt1:tt1_2_flex-tt1]

          if(tt1_2_flex_dif > 0):
            threshold = np.percentile(amplitude_envelope[tt1_1_flex_dif:-tt1_2_flex_dif],thresholdPercentile)
          else:
            threshold = np.percentile(amplitude_envelope[tt1_1_flex_dif:],thresholdPercentile)

          z = amplitude_envelope>threshold          ## 0 and 1 array corresponding to bursts vs non-bursts


          if(useHMM):

            ## HMM implemetation

            """ remodel = hmm.GMMHMM(n_components=2,covariance_type="full", n_iter=100000)
            remodel.fit(amplitude_envelope.reshape(-1, 1))
            proba = remodel.predict_proba(amplitude_envelope.reshape(-1, 1))

            ## setting state_prob == 0.66

            q =proba[:, 0]
            q[q > HMM_state_prob] = 1
            q[q < HMM_state_prob] = 0

            w=proba[:, 1]
            w[w > HMM_state_prob] = 1
            w[w < HMM_state_prob] = 0

            #diff = 0

            if np.mean(w!=z) > np.mean(q!=z):             ## determine whether the burst state is q or w based on which aligns better with the threshold method
              b_state = q
              #diff = q!=z
            else:
              b_state = w
              #diff = w!=z """


          else:

            z = np.array([float(x) for x in z])
            b_state = z


          #grouped_l = condense each burst and non-burst group to one value: 0,0,0,1,1,0,0,0,0 -> 0,1,0
          #grouped_l1 = save the index at the beginning of each burst/non-burst group: 0,0,0,1,1,0,0,0,0 -> 3,5,9
          #o = burst duration (how many time points it takes)
          #bamp = max burst amplitude
          #grouped_l3 = all burst durations within same subepoch
          #grouped_l4 = bamp_f_1 = all burst amplitudes within same subepoch



          j_1=0
          grouped_l= []
          grouped_l = [k for k,g in groupby(b_state)]            ## condense each burst and non-burst group to one value: 0,0,0,1,1,0,0,0,0 -> 0,1,0

          grouped_l1= []
          grouped_l3= []
          grouped_l4= []
          for k,g in groupby(b_state):
            j_1 = j_1 +sum(1 for i in g)
            grouped_l1.append (j_1)                              ## save the index at the beginning of each burst/non-burst group: 0,0,0,1,1,0,0,0,0 -> 3,5,9

          o=0
          bamp=0
          bamp_f_1 =0

          if grouped_l[0]==1:                                   ## if the first instance/group is a burst
            o=0
            bamp =0
            o= sum(b_state[0:grouped_l1[0]])                  ## burst duration (how many time points it takes)

            bamp = bamp+ max(amplitude_envelope[0:grouped_l1[0]])

            if o > min_burst_duration and grouped_l1[0] >= tt1_1_flex_dif:   

              grouped_l3.append(o)
              grouped_l4.append(bamp)
            for i in range(1,len(grouped_l1)-1,2):
              o = 0
              bamp = 0
              o = o+ sum(b_state[grouped_l1[i]:grouped_l1[i+1]])
              bamp = bamp+ max(amplitude_envelope[grouped_l1[i]:grouped_l1[i+1]])

              # for flexible intervals - check whether the burst starts or ends within the interval (if it ends after the beginning of the interval AND it starts before the end of the interval)
              if o > min_burst_duration and grouped_l1[i+1] >= tt1_1_flex_dif and grouped_l1[i] + tt1_1_flex <= tt1_2:
                grouped_l3.append(o)
                grouped_l4.append(bamp)
              o = 0

          else:                                                       ## if the first instance/group is a non-burst
            for i in range(0,len(grouped_l1)-1,2):
              o = 0
              bamp = 0
              o = o+ sum(b_state[grouped_l1[i]:grouped_l1[i+1]])
              bamp = bamp+ max(amplitude_envelope[grouped_l1[i]:grouped_l1[i+1]])

              if o > min_burst_duration and grouped_l1[i+1] >= tt1_1_flex_dif and grouped_l1[i] + tt1_1_flex <= tt1_2:
                grouped_l3.append(o)
                grouped_l4.append(bamp)

              o=0

          if sum(grouped_l3) ==0:
            bamp_f_1 = 0
            retry += 1
          else:

            bamp_f_1 =  grouped_l4
            break


        for i in range(len(grouped_l3)):  # for each burst within the same segment
          res1.append([])
          res1[-1] = [channels.iloc[ch] + "_" + epochs_name[sub_epoch] + "_" + str(tt) + "_" + str(i)]
          res1[-1].append(len(grouped_l3)/interval)        ## rate = number of bursts / time_interval
          res1[-1].append(bamp_f_1[i])               ## ampl

          res1[-1].append(grouped_l3[i]*burst_duration_in_ms_conversion)  ## duration

        if len(grouped_l3) == 0:      # if no bursts are present, fill with 0s
          res1.append([])
          res1[-1] = [channels.iloc[ch] + "_" + epochs_name[sub_epoch] + "_" + str(tt) + "_" + str(0)]
          res1[-1].append(0)        ## rate = number of bursts / time_interval
          res1[-1].append(0)               ## ampl

          res1[-1].append(0)  ## duration


  print(imp) # track progress of burst extraction in the log
  
  np.savetxt(exportFile, res1, delimiter=",", fmt='%s')



