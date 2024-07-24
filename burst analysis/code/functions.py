from scipy import stats
#import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from os.path import isfile, join
from decimal import Decimal
from scipy.stats import sem




###
# participant class containing:

# dict of task
# rest =  dict of session -> dict of block -> dict of channels -> trial class
# motor =  dict of session -> dict of block -> dict of channels -> motor class

# motor class - prop: dict of mov_interval (pre, mov, post) = array of trial class 

# trial class - prop: rate, array of burst class
# burst class - prop: ampl, duration


# participant (e.g. AD0109), task (rest vs motor), session (20Hz vs 70Hz vs sham), block (base, post15, post45)
# channel (8 channels), trials (1 vs variable for motor), 
# mov interval (rest vs pre, mrbd, pmbr), bursts (variable on the number of them), burst characteristics (rate, ampl, duration)


# filename = bursts_participant_task_session_block.csv
# e.g. bursts_AD0109_motor_20Hz_post15.csv
# e.g. bursts_AD0109_rest_sham_base.csv

###

class Burst:
    def __init__(self, ampl, duration):
        self.ampl = ampl
        self.duration = duration

class Trial:
    def __init__(self, rate, bursts):
        self.rate = rate
        self.bursts = bursts

class Motor:
    def __init__(self, mov_interval):
        self.mov_interval = mov_interval

class Participant:
    def __init__(self, id, task):
        self.id = id
        self.task = task


def import_all_files(path, bad_channels_file, channels, motor_intervals_of_interest, rest_intervals_of_interest):

    importFiles = [f for f in os.listdir(path) if isfile(join(path, f))]

    participants = []

    for f in importFiles:

        file_split = f.split(".")
        file_split = file_split[0].split("_")

        channels = prepare_file_bursts(join(path, f), bad_channels_file, channels, motor_intervals_of_interest, rest_intervals_of_interest, task = file_split[2])

        

        participant = None

        for p in participants:
            if p.id == file_split[1]:
                participant = p
                break

        if participant is None:
            participant = Participant(file_split[1], dict())
            participants.append(participant)
        
        if file_split[2] not in participant.task:
            participant.task[file_split[2]] = dict()

        if file_split[3] not in participant.task[file_split[2]]:
            participant.task[file_split[2]][file_split[3]] = dict()

        participant.task[file_split[2]][file_split[3]][file_split[4]] = channels

    return participants




def prepare_file_bursts(filename, bad_channels_file, channels, motor_intervals_of_interest, rest_intervals_of_interest, task = "motor", alt_channels=[]):




    channels_left = channels
    channels_right = alt_channels



    df = pd.read_csv (filename,header=None)
    df=df.loc[~(df==0).all(axis=1)] 

    df = df.to_numpy()
    trial_info = df[1:,0]
    df = df[1:,1:]
    
    channels = dict()

    trial = None




    bad_channels = extract_bad_channels(bad_channels_file)
    curr_file = filename.split("bursts_")[1]

    number_trials = 0
    for i in range(df.shape[0]):

        
        split_trial = trial_info[i].split("_") # channel, mov interval, trial number, burst number

        # determine the total number of trials
        if int(split_trial[2]) > number_trials:
            number_trials = int(split_trial[2])
        
        # check for bad channels
        if split_trial[0] in bad_channels[curr_file]:
             continue
        
        # transpose right electrodes onto left electrodes
        if split_trial[0] in channels_right:
            index = channels_right.index(split_trial[0])
            split_trial[0] = channels_left[index]


        if task == "motor":
            # only record intervals of interest
            if split_trial[1] not in motor_intervals_of_interest:
                continue

            if not split_trial[0] in channels:
                channels[split_trial[0]] = Motor(dict())
            
            if split_trial[1] not in channels[split_trial[0]].mov_interval:
                channels[split_trial[0]].mov_interval[split_trial[1]] = []

            if(split_trial[3] == "0"):

                if(float(df[i,0]) != 0):
                    trial = Trial(float(df[i,0]), [Burst(float(df[i,1]), float(df[i,2]))])
                else:
                    trial = Trial(float(df[i,0]), [])

                channels[split_trial[0]].mov_interval[split_trial[1]].append(trial)

            else:
                arr = channels[split_trial[0]].mov_interval[split_trial[1]]

                arr[arr.index(trial)].bursts.append(Burst(float(df[i,1]), float(df[i,2])))

        else:
            # only record intervals of interest
            if split_trial[1] not in rest_intervals_of_interest:
                continue
            
            if not split_trial[0] in channels:
                channels[split_trial[0]] = []

            if(split_trial[3] == "0"):
                if(float(df[i,0]) != 0):
                    trial = Trial(float(df[i,0]), [Burst(float(df[i,1]), float(df[i,2]))])
                else:
                    trial = Trial(float(df[i,0]), [])
                
                channels[split_trial[0]].append(trial)

            else:
                arr = channels[split_trial[0]]

                arr[arr.index(trial)].bursts.append(Burst(float(df[i,1]), float(df[i,2])))
            
    return channels



#### return out =  dict of channels -> dict session (sham, 20Hz, 70Hz) -> dict block (base, post15, post45) -> 
# -> mov_interval (rest, pre, mov, post) -> dict component (rate, ampl, dur) -> array

### return mrbd = dict of channels -> dict session (sham, 20Hz, 70Hz) -> dict block (base, post15, post45) -> 
# -> mov_interval (rest, pre, mov, post) -> dict component (rate, ampl, dur) 
# -> array of tuple (avg value per participant, number of elements averaged)
def extract_components(path, bad_channels_file, channels, motor_intervals_of_interest, rest_intervals_of_interest):
    participants = import_all_files(path, bad_channels_file, channels, motor_intervals_of_interest, rest_intervals_of_interest)

    out = dict()
    mrbd = dict()
    num_participants = len(participants)
    participant_index = -1
    for participant in participants:
        participant_index += 1
        for task_key, task_val in participant.task.items():
            for session_key, session_val in task_val.items():
                for block_key, block_val in session_val.items():
                    for channel_key, channel_val in block_val.items():

                        if channel_key not in out:
                            out[channel_key] = dict()
                            mrbd[channel_key] = dict()
                        if session_key not in out[channel_key]:
                            out[channel_key][session_key] = dict()
                            mrbd[channel_key][session_key] = dict()
                        if block_key not in out[channel_key][session_key]:
                            out[channel_key][session_key][block_key] = dict()
                            mrbd[channel_key][session_key][block_key] = dict()

                        if task_key == "motor":
                            for mov_interval_key, mov_interval_value in channel_val.mov_interval.items():

                                if mov_interval_key not in out[channel_key][session_key][block_key]:
                                    out[channel_key][session_key][block_key][mov_interval_key] = dict()
                                    mrbd[channel_key][session_key][block_key][mov_interval_key] = dict()

                                ## convert trial results into dict components 
                                trials_2_dict_components(mov_interval_value, out[channel_key][session_key][block_key][mov_interval_key], mrbd[channel_key][session_key][block_key][mov_interval_key], num_participants, participant_index)

                        else:
                            if task_key not in out[channel_key][session_key][block_key]:
                                out[channel_key][session_key][block_key][task_key] = dict()
                                mrbd[channel_key][session_key][block_key][task_key] = dict()

                            ## convert trial results into dict components 
                            trials_2_dict_components(channel_val, out[channel_key][session_key][block_key][task_key], mrbd[channel_key][session_key][block_key][task_key], num_participants, participant_index)
    
    return (out, mrbd)




# convert trial results into dict components 
def trials_2_dict_components(data, out, mrbd, num_participants, participant_index):

    if "rate" not in out:
        out["rate"] = []
        mrbd["rate"] = []
        for i in range(num_participants):
            mrbd["rate"].append([])
    if "amplitude" not in out:
        out["amplitude"] = []
        mrbd["amplitude"] = []
        for i in range(num_participants):
            mrbd["amplitude"].append([])
    if "duration" not in out:
        out["duration"] = []
        mrbd["duration"] = []
        for i in range(num_participants):
            mrbd["duration"].append([])

    for trial in data:
        out["rate"].append(trial.rate)

        if len(mrbd["rate"][participant_index]) == 0:
            mrbd["rate"][participant_index].append(trial.rate)
            mrbd["rate"][participant_index].append(1)

        else:
            n = mrbd["rate"][participant_index][1]
            val = mrbd["rate"][participant_index][0] 
            mrbd["rate"][participant_index][0] = (val * n + trial.rate) / (n+1)
            mrbd["rate"][participant_index][1] = n+1

        for burst in trial.bursts:
            out["amplitude"].append(burst.ampl)
            out["duration"].append(burst.duration)

            if len(mrbd["amplitude"][participant_index]) == 0:
                mrbd["amplitude"][participant_index].append(burst.ampl)
                mrbd["amplitude"][participant_index].append(1)
                mrbd["duration"][participant_index].append(burst.duration)
                mrbd["duration"][participant_index].append(1)

            else:
                n = mrbd["amplitude"][participant_index][1]
                val = mrbd["amplitude"][participant_index][0] 
                mrbd["amplitude"][participant_index][0] = (val * n + burst.ampl) / (n+1)

                val2 = mrbd["duration"][participant_index][0]
                mrbd["duration"][participant_index][0] = (val2 * n + burst.duration) / (n+1)

                mrbd["amplitude"][participant_index][1] = n+1
                mrbd["duration"][participant_index][1] = n+1




# save the data on bad channels from a csv file into a python array
def extract_bad_channels(path):

    df = pd.read_csv (path,header=None, engine='python')

    df=df.loc[~(df==0).all(axis=1)]

    bad_ch_arr = np.array(df)

    bad_channels = dict()

    for i in range(len(bad_ch_arr)):
        bad_channels[bad_ch_arr[i,0]] = []

        ch = bad_ch_arr[i,1].split(", ")
        for j in ch:
            if j == "N":
                break
            bad_channels[bad_ch_arr[i,0]].append(j)
    
    return bad_channels



# function to plot the MRBD used on Power Analysis (NOT beta bursts)
def plot_mrbd(filename):
    df = pd.read_csv (filename,header=None)
    df=df.loc[~(df==0).all(axis=1)] 

    if(np.isnan(df.iloc[-1,-1])):
        df = df.iloc[:,:-1]

    channels = df.iloc[1:,0]
    time_points = df.iloc[0, 1:]
    df = df.iloc[1:,1:]
    
    
    i = list(channels).index('C3')

    plt.plot(time_points[20:-20],df.iloc[i,20:-20])
    plt.axhline(y=0, c='r', ls='--', lw='.5')

    plt.xlabel('Time (s)', fontsize=15)
    plt.ylabel('Relative power', fontsize=15)

    plt.suptitle('Baseline MRBD - C3 electrode', fontsize=15)
        

    plt.show()




# Line Plots with error bars and calculates t-tests between each point
# data = [[[xTick1_group1], [xTick2_group1], ...], [[xTick1_group2], [xTick2_group2], ...], ...]

def line_error(data, title, ylabel, xticklabels, legend, electrodes, normalize = True, bonferroni = True):

    ax = plt.axes()
    ax.tick_params(axis='x', which='major', labelsize=15)
    ax.set_ylabel(ylabel, fontsize=15)

    
    
    #### Different color schemes
    colors = sns.color_palette('pastel').as_hex()    
    #colors = ['#1a275e', '#bd328c', '#fd855a']
    #colors = ['#1d94bb', '#ee1404', '#ffc000']

    data_el = data

    for i, legend_group in enumerate(data_el):

        points = []
        err = []

        normalization_value = np.average(legend_group[0])
        for j, xtick_data in enumerate(legend_group):

            if(normalize):
                arr = [x-normalization_value for x in np.array(xtick_data)]
            else:
                arr = np.array(xtick_data)

            legend_group[j] = arr
            points.append(np.average(arr))
            err.append(sem(arr))
        
        ax.errorbar(xticklabels, points, yerr=err, capsize=5, c = colors[i], lw = 2.5)

    
    ax.legend(legend, loc="upper left")
    

    for group in range(len(legend)):
        # Get info about y-axis
        bottom, top = ax.get_ylim()
        yrange = top - bottom

        # Check for statistical significance
        significant_combinations = []
        # Check from the outside pairs of boxes inwards
        ls = list(range(1, len(data_el[group]) + 1))
        combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]

        correction = 1

        if (bonferroni):
            correction = len(combinations) * len(legend) * 3

        for c in combinations:
            data1 = data_el[group][c[0] - 1]
            data2 = data_el[group][c[1] - 1]
            # Significance
            U, p = stats.ttest_ind(data1, data2, alternative='two-sided')
            if p < 0.05 / correction:
                significant_combinations.append([c, p])

        # Significance bars
        for i, significant_combination in enumerate(significant_combinations):
            # Columns corresponding to the datasets of interest
            x1 = significant_combination[0][0] - 1
            x2 = significant_combination[0][1] - 1
            # What level is this bar among the bars above the plot?
            level = len(significant_combinations) - i
            # Plot the bar
            bar_height = (yrange * 0.11 * level) + top
            bar_tips = bar_height - (yrange * 0.02)
            ax.plot(
                [x1, x1, x2, x2],
                [bar_tips, bar_height, bar_height, bar_tips], lw=1.5, c=colors[group])

            # Significance level
            p = significant_combination[1]
            if p < 0.001 / correction:
                sig_symbol = '***'
            elif p < 0.01 / correction:
                sig_symbol = '**'
            elif p < 0.05 / correction:
                sig_symbol = '*'
            text_height = bar_height + (yrange * 0.01)
            ax.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c=colors[group], size = 'large')


    # Get info about y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom
    
    ax.set_ylim(bottom, top + yrange*0.04)
    ax.xaxis.label.set_size(20)

    plt.show()



# Unused lately... might require some adjustments if you need to use it

# input data: [[[xTick1_group1], [xTick1_group2], ...], [[xTick2_group1], [xTick2_group2], ...], ...]
def box_and_whisker(input_data, title, ylabel, xticklabels, bonferroni = False, show_avg  = True, legend = []):
    """
    Create a box-and-whisker plot with significance bars.
    """

    if type(input_data[0][0]) is list or type(input_data[0][0]) is np.ndarray:
        data = input_data
    else:
        data = [input_data]

    num_blocks = len(data)
    group_size = len(data[0])

    ax = plt.axes()
    # Graph title
    ax.set_title(title, fontsize=14)
    # Label y-axis
    ax.set_ylabel(ylabel)

    for group in range(num_blocks):
        bp = ax.boxplot(data[group], positions = [(x + group*(group_size + 1) + 1) for x in range(group_size)], widths=0.6, patch_artist=True)
        
        # Change the colour of the boxes to Seaborn's 'pastel' palette
        colors = sns.color_palette('pastel')
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        # Colour of the median lines
        plt.setp(bp['medians'], color='k')


    # Hide x-axis major ticks
    ax.tick_params(axis='x', which='major', length=0)
    # Show x-axis minor ticks
    if num_blocks == 1:
        xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
    else:
        xticks = [x*(group_size+1) for x in range(num_blocks+1)]
    ax.set_xticks(xticks, minor=True)
    # Clean up the appearance
    ax.tick_params(axis='x', which='minor', length=3, width=1)

    # Label x-axis ticks
    if num_blocks > 1:
        xlabel_positions = [x + (group_size + 1)/2 for x in xticks]
        xlabel_positions = xlabel_positions[:-1]
        ax.set_xticks(xlabel_positions, minor=False)
    ax.set_xticklabels(xticklabels)
    
    # Get info about y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom

    

    for group in range(num_blocks):

        # Check for statistical significance
        significant_combinations = []
        # Check from the outside pairs of boxes inwards
        ls = list(range(1, len(data[group]) + 1))
        combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]

        correction = 1

        if (bonferroni):
            correction = len(combinations) * num_blocks

        for c in combinations:
            data1 = data[group][c[0] - 1]
            data2 = data[group][c[1] - 1]
            # Significance
            #U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            U, p = stats.ttest_ind(data1, data2, alternative='two-sided')
            if p < 0.05 / correction:
                significant_combinations.append([c, p])

        # Significance bars
        for i, significant_combination in enumerate(significant_combinations):
            # Columns corresponding to the datasets of interest
            x1 = significant_combination[0][0] + group*(group_size+1)
            x2 = significant_combination[0][1] + group*(group_size+1)
            # What level is this bar among the bars above the plot?
            level = len(significant_combinations) - i
            # Plot the bar
            bar_height = (yrange * 0.08 * level) + top
            bar_tips = bar_height - (yrange * 0.02)
            plt.plot(
                [x1, x1, x2, x2],
                [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')

            # Significance level
            p = significant_combination[1]
            if p < 0.001 / correction:
                sig_symbol = '***'
            elif p < 0.01 / correction:
                sig_symbol = '**'
            elif p < 0.05 / correction:
                sig_symbol = '*'
            text_height = bar_height + (yrange * 0.01)
            plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

    # Get info about y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom

    if(show_avg):
        ax.set_ylim(bottom - 0.045 * yrange, top)
    else:
        ax.set_ylim(bottom - 0.02 * yrange, top)
    bottom_avg = bottom - 0.025 * yrange

    for group in range(num_blocks):
        avg = []
        for i in range(len(data[group])):
            avg.append(sum(data[group][i])/len(data[group][i]))
        
        # Annotate sample size below each box
        for i, dataset in enumerate(data[group]):
            sample_size = len(dataset)
            x_position = (i+1) + group*(group_size+1)
            ax.text(x_position, bottom, fr'n = {sample_size}', ha='center', size='x-small')

            if(show_avg):
                if (avg[0] > 0.1 or avg[0] < -0.1):
                    ax.text(x_position, bottom_avg, fr'avg = {"{:.2f}".format(avg[i])}', ha='center', size='x-small')
                else:
                    ax.text(x_position, bottom_avg, fr'avg = {"%.2E" % Decimal(avg[i])}', ha='center', size='x-small')


    # draw temporary red and blue lines and use them to create a legend
    if num_blocks > 1:
        colors = sns.color_palette('pastel').as_hex()

        temp_legend_lines = []
        for i in range(group_size):
            hB, = ax.plot([3,3], colors[i])
            temp_legend_lines.append(hB)
        temp_legend_lines = tuple(temp_legend_lines)
        legend = tuple(legend)


        ax.legend(temp_legend_lines, legend, loc='upper left')

        for e in temp_legend_lines:
            e.set_visible(False)
        
        left, right = ax.get_xlim()
        xrange = right - left

        ax.set_xlim(left, (group_size+1)*num_blocks + 0.04 * xrange)
    #hR, = ax.plot([1,1], colors[1])
    
    # ax.legend((hB, hR),legend)
    # hB.set_visible(False)
    # hR.set_visible(False)
    plt.show()








# def old_extract_components(path, bad_channels_file, channel = "C1"):
#     participants = import_all_files(path, bad_channels_file)

#     ## rate array [[pre][mov][post]]
#     ## rate dict of arrays 
#     rate = dict()
#     ampl = dict()
#     duration = dict()

#     for participant in participants:
#         for session in participant.task["rest"]:
#             for channels in participant.task["rest"][session]:
#                 channels = participant.task["rest"][session][channels]

#                 for mov_interval_key, mov_interval_value in channels[channel].mov_interval.items():
                    
#                     if mov_interval_key not in rate:
#                         rate[mov_interval_key] = []
#                         ampl[mov_interval_key] = []
#                         duration[mov_interval_key] = []

#                     for trial in mov_interval_value:
#                         rate[mov_interval_key].append(trial.rate)
#                         for burst in trial.bursts:
#                             ampl[mov_interval_key].append(burst.ampl)
#                             duration[mov_interval_key].append(burst.duration)

#                 """ for trial in channels[channel].mov:
#                     rate[1].append(trial.rate)
#                     for burst in trial.bursts:
#                         ampl[1].append(burst.ampl)
#                         duration[1].append(burst.duration)
                
#                 for trial in channels[channel].post:
#                     rate[2].append(trial.rate)
#                     for burst in trial.bursts:
#                         ampl[2].append(burst.ampl)
#                         duration[2].append(burst.duration) """
    
#     return (rate, ampl, duration)



# def plot_mrbd_old(filename):
#     df = pd.read_csv (filename,header=None)
#     df=df.loc[~(df==0).all(axis=1)] 

#     if(np.isnan(df.iloc[-1,-1])):
#         df = df.iloc[:,:-1]

#     channels = df.iloc[1:,0]
#     time_points = df.iloc[0, 1:]
#     df = df.iloc[1:,1:]

#     fig, axs = plt.subplots(2,4)
    
    
#     for i in range(8):
#         ax = axs.flat[i]
#         ax.plot(time_points[20:-20],df.iloc[i,20:-20])
#         ax.axhline(y=0, c='r', ls='--', lw='.5')

#         ax.set_ylim((-45,40))

#         ax.set_title(channels[i+1])
#         ax.set(xlabel='Time (s)', ylabel='Relative power')
#         ax.label_outer()

#         fig.suptitle('Baseline MRBD')
        
#     plt.show()