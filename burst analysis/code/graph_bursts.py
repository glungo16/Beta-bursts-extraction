import numpy as np
import pandas as pd
from functions import *
from scipy.stats import sem
import sys


#### TO WORK:
# requires functions.py file in the same directory
# requires a bad_channels_file



######## PARAMETERS

bad_channels_file = r"C:\Cristi\Grad McGill\Project\Beta bursts\data_extracted\bad_channels\bad_channels.csv"

# path of the input folder
path = r"C:\Cristi\Grad McGill\Project\Beta bursts\data_extracted\filtering_each_epoch\flex_int_03"


channels = ['FC5', 'FC3', 'FC1', 'C5', 'C3', 'C1', 'CP5', 'CP3', 'CP1']


# only save the intervals of interest and removes all the extra segments (e.g. removes small segments used when calculating flexible intervals)
motor_intervals_of_interest = ['pre', 'mov', 'post']
rest_intervals_of_interest = ['int']


######### END PARAMETERS


(data, mrbd) = extract_components(path, bad_channels_file, channels, motor_intervals_of_interest, rest_intervals_of_interest)


electrodes = channels
# electrodes array not used in line_error() function since I only needed to plot for C3
# Adjustments are needed to the function if you want to plot all electrodes in the same graph

###### Plots


#### Resting data
## Compare sessions
# xticklables = block (base, post15, post45)
# legend = session (sham, 20Hz, 70Hz)



xticklabels = []
legend_group = []

myData = dict()

xtl_index = -1
legend_index = -1

for electrode_key, electrode_val in data.items():
    if electrode_key != 'C3' and electrode_key != 'C4':
        continue
    for session_key, session_val in electrode_val.items():

        ## Legend change
        legend_group = list(electrode_val.keys())
        legend_index = legend_group.index(session_key)

        for block_key, block_val in session_val.items():

            ## Xticks change
            xticklabels = list(session_val.keys())
            xtl_index = xticklabels.index(block_key)

            for comp_key, comp_val in block_val['rest'].items():
                
                if comp_key not in myData:
                    myData[comp_key] = []
                    for i in range(len(legend_group)):
                        myData[comp_key].append([])
                        for j in range(len(xticklabels)):
                            myData[comp_key][i].append([])


                myData[comp_key][legend_index][xtl_index] = comp_val


title = "Resting state data"
normalization = True
xticklabels = ['Baseline', 'Post-15 min', 'Post-45 min']


# Comment/Uncomment each section depending on what you want to plot at the moment
ylabel = "Burst Rate"
line_error(myData['rate'], title, ylabel, xticklabels, legend_group, electrodes, bonferroni = True, normalize=normalization)

ylabel = u"Burst Amplitude"
line_error(myData['amplitude'], title, ylabel, xticklabels, legend_group, electrodes, bonferroni = True, normalize=normalization)

ylabel = "Burst Duration"
line_error(myData['duration'], title, ylabel, xticklabels, legend_group, electrodes, bonferroni = True, normalize=normalization)




#### Motor data

### For each session (sham, 20Hz, 70Hz)

## Compare mov intervals (rest, pre, mov, post)
# xticklables = mov interval (rest, pre, mov, post)
# legend = block (base, post15, post45)

session = 'sham'

xticklabels = []
legend_group = []

myData = dict()

xtl_index = -1
legend_index = -1

        

for electrode_key, electrode_val in data.items():
    if electrode_key != 'C3':
        continue

    for block_key, block_val in electrode_val[session].items():
        
        ## Legend change
        legend_group = list(electrode_val[session].keys())
        legend_index = legend_group.index(block_key)

        for interval_key, interval_val in block_val.items():
            
            if interval_key == "rest":
                continue
            ## Xticks change
            xticklabels = list(block_val.keys())
            xtl_index = xticklabels.index(interval_key)

            for comp_key, comp_val in interval_val.items():
                
                if comp_key not in myData:
                    myData[comp_key] = []
                    for i in range(len(legend_group)):
                        myData[comp_key].append([])
                        for j in range(len(xticklabels)):
                            myData[comp_key][i].append([])



                myData[comp_key][legend_index][xtl_index] = comp_val


title = "Motor data - " + session + " session"
normalization = True
xticklabels = ['Pre-movement', 'Movement', 'Post-movement', '']

# Comment/Uncomment each section depending on what you want to plot at the moment
ylabel = "Burst Rate"
#line_error(myData['rate'], title, ylabel, xticklabels, legend_group, electrodes, bonferroni = True, normalize=normalization)

ylabel = u"Burst Amplitude"
#line_error(myData['amplitude'], title, ylabel, xticklabels, legend_group, electrodes, bonferroni = True, normalize=normalization)

ylabel = "Burst Duration"
#line_error(myData['duration'], title, ylabel, xticklabels, legend_group, electrodes, bonferroni = True, normalize=normalization)


### For each session (sham, 20Hz, 70Hz)

## Compare blocks (base, post15, post45)
# xticklables = block (base, post15, post45)
# legend = mov interval (rest, pre, mov, post)


session = 'tDCS'

xticklabels = []
legend_group = []

myData = dict()

xtl_index = -1
legend_index = -1

        

for electrode_key, electrode_val in data.items():
    if electrode_key != 'C3':
        continue
    for block_key, block_val in electrode_val[session].items():
        
        
        ## Xticks change
        xticklabels = list(electrode_val[session].keys())
        xtl_index = xticklabels.index(block_key)

        for interval_key, interval_val in block_val.items():

            if interval_key == "rest":
                continue
            ## Legend change
            legend_group = list(block_val.keys())
            legend_index = legend_group.index(interval_key)

            for comp_key, comp_val in interval_val.items():
                
                if comp_key not in myData:
                    myData[comp_key] = []
                    for i in range(len(legend_group) - 1):
                        myData[comp_key].append([])
                        for j in range(len(xticklabels)):
                            myData[comp_key][i].append([])


                myData[comp_key][legend_index][xtl_index] = comp_val


title = "Motor data - " + session + " session"
normalization = True
legend_group = ['Pre-movement', 'Movement', 'Post-movement']
xticklabels = ['Baseline', 'Post-15 min', 'Post-45 min']

# Comment/Uncomment each section depending on what you want to plot at the moment
ylabel = "Burst Rate"
#line_error(myData['rate'], title, ylabel, xticklabels, legend_group, electrodes, bonferroni = True, normalize=normalization)

ylabel = u"Burst Amplitude"
#line_error(myData['amplitude'], title, ylabel, xticklabels, legend_group, electrodes, bonferroni = True, normalize=normalization)

ylabel = "Burst Duration"
#line_error(myData['duration'], title, ylabel, xticklabels, legend_group, electrodes, bonferroni = True, normalize=normalization)





#### MRBD data

# MRBD calculated from first resting baseline period

# MRBD% = (C(t) - B) / B * 100%

## xticklabels = block (base, post15, post45)
## legend = session (sham, 20Hz, 70Hz)

xticklabels = []
legend_group = []

myData = dict()

xtl_index = -1
legend_index = -1

for electrode_key, electrode_val in data.items():
    if electrode_key != 'C3':
        continue
    for session_key, session_val in electrode_val.items():

        ## Legend change
        legend_group = list(electrode_val.keys())
        legend_index = legend_group.index(session_key)

        for block_key, block_val in session_val.items():
            
            
            ## Xticks change
            xticklabels = list(session_val.keys())
            xtl_index = xticklabels.index(block_key)


            for comp_key, comp_val in block_val['mov'].items():
                
                if comp_key not in myData:
                    myData[comp_key] = []
                    for i in range(len(legend_group)):
                        myData[comp_key].append([])
                        for j in range(len(xticklabels)):
                            myData[comp_key][i].append([])

                    
                C = comp_val
                B = sum(session_val['base']['rest'][comp_key])/len(session_val['base']['rest'][comp_key])
                
                MRBD = []
                for c in C:
                    MRBD.append((c - B) / B * 100)

                myData[comp_key][legend_index][xtl_index] = MRBD


title = "MRBD data"
normalization = True
xticklabels = ['Baseline', 'Post-15 min', 'Post-45 min']

# Comment/Uncomment each section depending on what you want to plot at the moment
ylabel = "MRBD Burst Rate (%)"
#ine_error(myData['rate'], title, ylabel, xticklabels, legend_group, electrodes, bonferroni = True, normalize=normalization)

ylabel = u"MRBD Burst Amplitude (%)"
#ine_error(myData['amplitude'], title, ylabel, xticklabels, legend_group, electrodes, bonferroni = True, normalize=normalization)

ylabel = "MRBD Burst Duration (%)"
#line_error(myData['duration'], title, ylabel, xticklabels, legend_group, electrodes, bonferroni = True, normalize=normalization)


