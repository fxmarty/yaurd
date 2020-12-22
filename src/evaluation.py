import os

import torch
import numpy as np

import librosa
import soundfile as sf

from unvoiced_detection import detect

import pandas as pd
import bisect

import matplotlib.pyplot as plt
import seaborn as sns

import argparse

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

if __name__ == '__main__':
    # Parser settings
    parser = argparse.ArgumentParser(
        description='Parser',
        add_help=False
    )

    parser.add_argument(
        '--nus-path',
        type=str,
        required=True,
        help='path to NUS-48E folder'
    )
    
    parser.add_argument(
        '--reading',
        action='store_true',
        default=False,
        help='path to NUS-48E folder'
    )
    
    parser.add_argument('--out-path',type=str,required=True,
                        default=None, help='Path to an evaluation directory')
        
    args, _ = parser.parse_known_args()
    print("args out path",args.out_path)
    # recreate NUS-48E structure for evaluation
    for dirpath, dirnames, filenames in os.walk(args.nus_path):
        structure = os.path.join(args.out_path, dirpath[len(args.nus_path)+1:])
        if not os.path.isdir(structure):
            os.mkdir(structure)
        
    
    if args.reading == True:
        subdir_name = 'read'
    else:
        subdir_name = 'sing'
    
    unvoiced_phonemes = [
                    'ch',
                    'dh',
                    'f',
                    'hh',
                    'k',
                    'p',
                    's',
                    'sh',
                    't',
                    'th',
                    # to discuss
                    'sil', # allowed because the singer inhaling is labeled as
                           # silence in NUS-48E
                    'jh',
                    'sp' # found 's' labeled as 'sp' instead
                ]
    
    directory_list = get_immediate_subdirectories(args.nus_path)
        
    metrics_overall = []
    for dir_song in directory_list:
        print("Singer",dir_song)
        extract_names = []
        for file in os.listdir(os.path.join(args.nus_path,dir_song,subdir_name)):
            if file.endswith(".txt"):
                extract_names.append(os.path.splitext(file)[0])

        for extract in extract_names:
            audio, rate = sf.read(
                os.path.join(args.nus_path,dir_song,subdir_name,extract + '.wav'),
                always_2d=False
            ) # audio is a numpy array with size (nb_timesteps, nb_channels)
        
            if len(audio.shape) == 1:
                audio_mono = audio
            
            elif audio.shape[1] == 2:
                audio_mono = audio.sum(axis=1) / 2
            
            # estimates
            est_unv_start,est_unv_end = detect(audio_mono,rate)
            
            reference_file_path = os.path.join(args.nus_path,dir_song,
                                                subdir_name,extract + '.txt')
            df = pd.read_csv(reference_file_path,delim_whitespace=True,header=None)  
            reference = df.to_numpy()
            
            
            metrics = []
            for i in range(len(est_unv_start)):
                est_start_sec = est_unv_start[i]/rate
                est_end_sec = est_unv_end[i]/rate
                closest_start_left_index = bisect.bisect_left(
                                            reference[:,0], est_start_sec) - 1
                closest_end_right_index = bisect.bisect_left(
                                            reference[:,1], est_end_sec)
                
                duration = est_end_sec - est_start_sec
                additive_correct = 0
                """
                print("est_start_sec",est_start_sec)
                print("est_end_sec",est_end_sec)
                print("closest_start_left_index",closest_start_left_index)
                print("closest_end_right_index",closest_end_right_index)
                print(reference[closest_start_left_index:closest_end_right_index+1])
                """
                
                # to avoid some pieces wrongly labeled in the dataset
                # (e.g. ADIZ 13.txt at 71.625985)
                good_labels = True
                for j in range(closest_start_left_index,closest_end_right_index+1):
                    if reference[j,1] != reference[j+1,0]:
                        good_labels = False
                
                if good_labels: # only rate the extract if the labels of the dataset
                                # are good
                    if closest_start_left_index == closest_end_right_index:
                        additive_correct += ((reference[closest_start_left_index,2] 
                                                in unvoiced_phonemes) * 
                                                (est_end_sec - est_start_sec))
                    else:
                        for j in range(closest_start_left_index,
                                        closest_end_right_index+1):
                            if j == closest_start_left_index:
                                additive_correct += ((reference[j,2] in 
                                                                    unvoiced_phonemes) 
                                                    * (reference[j,1] - est_start_sec))
                            
                            elif j == closest_end_right_index:
                                additive_correct += ((reference[j,2] in 
                                                                    unvoiced_phonemes)
                                                    * (est_end_sec - reference[j,0]))
                            
                            else:
                                additive_correct += ((reference[j,2] in 
                                                                    unvoiced_phonemes)
                                                    * (reference[j,1] - reference[j,0]))
                        
                    correctness = additive_correct/duration
                    metrics.append(correctness)
                    
                    if correctness < 0.3:
                        print("est_start_sec",est_start_sec)
                        print("est_end_sec",est_end_sec)
                        print("closest_start_left_index",closest_start_left_index)
                        print("closest_end_right_index",closest_end_right_index)
                        print(reference[closest_start_left_index:closest_end_right_index+1])
                        print("correctness",correctness)
                        print("----")
                    #print("correctness",correctness)
                    #print("----")
            
            out_file = os.path.join(args.out_path,dir_song,subdir_name,extract+'.txt')
            with open(out_file, 'w') as f:
                for item in metrics:
                    f.write("%s\n" % item)
            #print(metrics)
            metrics_overall.append(metrics)
    
    metrics_overall_flat = [item for sublist in metrics_overall for item in sublist]
    metrics_overall_flat = np.array(metrics_overall_flat)
    #print(metrics_overall)
    
    # 0.5 arbitrary threshold
    nb_good_detections = np.sum(metrics_overall_flat > 0.3)
    print("Overall, yaurd has",nb_good_detections,
            "good detections out of " + str(len(metrics_overall_flat)) +",",
            "which is",100*nb_good_detections/len(metrics_overall_flat),"%.")
    
    metrics_overall_flat_pd = pd.DataFrame(metrics_overall_flat,columns=["yaurd"])
    
    plt.figure()
    """
    sns.set_style("whitegrid")
    showfliers = False
    showmeans = True
    sns.boxplot(
                x='Method',
                y='value',
                data=pd.melt(metrics_overall_flat_pd,var_name='Method',
                            value_name='value'),
                whis=[10, 90],
                flierprops = dict(markerfacecolor = '0.50',
                                markersize = 4,marker='x'),
                showfliers=showfliers,
                showmeans=showmeans,
                meanprops={"marker":"d",
                    "markerfacecolor":"red", 
                    "markeredgecolor":"black",
                    "markersize":"5"},
                width=0.8,
            )
    
    sns.stripplot(
                x='Method',
                y='value',
                data=pd.melt(metrics_overall_flat_pd,var_name='Method',
                                value_name='value'),
                color='black',
                size=3
            )
    """
    nb_bins = 50
    bins=np.histogram(metrics_overall_flat, bins=nb_bins)[1] #get the bin edges
    plt.hist(metrics_overall_flat,bins=bins,edgecolor='k')
    
    plt.tight_layout()
    plt.grid()
    
    plt.title('Boxplot of the evaluation metrics',fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_path,'boxplot.png'), dpi=300)
    