import argparse
import os

import torch
import numpy as np

import librosa
import soundfile as sf

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import sklearn
import sklearn.mixture
import sklearn.covariance
from scipy.signal import argrelextrema
import scipy

import matplotlib
import csv
import ntpath

def detect(audio_mono,rate):
    """
    Input:
        audio_mono : numpy.array
            vocals audio, mono
        rate : int
            sampling rate
    Output:
        - unv_start : list
            list of starting points for the unvoiced sections
        - unv_end : list
            list of ending points for the corresponding unvoiced sections
    """
    
    start = 0
    end = audio_mono.shape[0]/rate

    excerpt = audio_mono[int(start*rate):int(end*rate)]
    x = np.linspace(start,end,excerpt.shape[0]) 
    
    # Zero Crossing Rate (ZCR) is the first feature to detect unvoiced sections.
    # High ZCR = unvoiced
    zcr = librosa.feature.zero_crossing_rate(excerpt,frame_length=int(0.02*rate),hop_length=int(0.02*rate)//4)
    x_zcr = np.linspace(start,end,zcr.shape[1])

    f, t, Zxx = scipy.signal.stft(excerpt, fs=rate,nperseg=int(0.04*rate),noverlap=3*int(0.04*rate)//4)
    
    Zxx = np.abs(Zxx)
    
    # We add this to have no error at normalization just after
    Zxx[0] = Zxx[0] + 0.00001

    Zxx = Zxx/Zxx.sum(axis=0)
    
    # The "proportion" of energy at high frequencies (> 3500 Hz) is our 2nd feature
    res = []
    for k in Zxx.T:
        res.append(k[f > 3500].sum())
    
    # Up sample to the same dimension as the feature for ZCR
    f = scipy.interpolate.interp1d(np.linspace(start, end,len(res)), res)
    res_resampled = f(x_zcr)
    
    # Combine the two features, normalize for k-means later on
    combined = np.vstack((zcr[0]/np.max(np.abs(zcr[0])), res_resampled/np.max(res_resampled))).T
    
    # Detect the frames for "complete" or near complete silence, in order for them
    # not to mess up k-means computation (they add a lot of weight to a single point
    # very close to 0, which is unwanted)
    indexes = []
    zero_values = []
    for i,j in enumerate(combined):
        if not (combined[i][0] > 0.01 and combined[i][1] > 0.01):
            indexes.append(i)

    combined = combined[(combined.T[0] > 0.01) & (combined.T[1] > 0.01)]
    
    nb_clusters=3
    
    kmeans = sklearn.cluster.KMeans(n_clusters=nb_clusters,init='k-means++',random_state=10).fit(combined)
    y_pred = kmeans.predict(combined)
    
    y_pred = list(y_pred)
    
    # Insert a 4th class for the complete silences, reintroduced in order.
    for k in indexes:
        y_pred.insert(k,nb_clusters)
    
    y_pred = np.array(y_pred)


    # Upsample to the original signal resolution
    f = scipy.interpolate.interp1d(x_zcr, y_pred,kind='nearest')
    class_resampled = np.int32(f(x))
    
    # From the 3 clusters from k-means, detect the index corresponding to each
    # of the 3 classes:
    #     - Voiced frames
    #     - Unvoiced frames
    #     - "Unknown" frames, in the middle
    index_voiced = kmeans.cluster_centers_.mean(axis=1).argmin()
    index_unvoiced = kmeans.cluster_centers_.mean(axis=1).argmax()
    
    list_index = [0,1,2]
    
    for i in sorted([index_voiced,index_unvoiced], reverse=True):
        del list_index[i]
    index_unknown = list_index[0]
    
    
    # Re-label the "unknown" frames surrounding an
    # unvoiced section to unvoiced IF their energy is <1.5 lower.
    current_type = class_resampled[0]
    current_start = 0
    past_start = -1
    past_end = -1
    past_type = -1
    
    for i,type in enumerate(class_resampled):
        if type == current_type:
            continue
        else:
            if past_type == index_unknown and current_type == index_unvoiced:
                if np.sum(excerpt[past_start:past_end]**2)/(past_end-past_start) < 1.5*np.sum(excerpt[current_start:i]**2)/(i-current_start):
                    class_resampled[past_start:past_end+1] = index_unvoiced
            
            if past_type == index_unvoiced and current_type == index_unknown:
                if np.sum(1.5*excerpt[past_start:past_end]**2)/(past_end-past_start) > np.sum(excerpt[current_start:i]**2)/(i-current_start):
                    class_resampled[current_start:i] = index_unvoiced
            
            past_type = current_type
            current_type = type
            past_start = current_start
            current_start = i
            past_end = i - 1
    
    
    # Compute the energy for each unvoiced section, in order later on to filter 
    # outliers with an elliptic envelope to refine the detection
    energy_unvoiced = []
    current_type = class_resampled[0]
    current_start = 0
    past_start = -1
    past_end = -1
    past_type = -1
    
    for i,type in enumerate(class_resampled):
        if type == current_type:
            continue
        else:
            if current_type == index_unvoiced:
                energy = np.sum(excerpt[current_start:i]**2)/(i-current_start)
                if energy != 0:
                    energy_unvoiced.append(energy)
                else:
                    energy_unvoiced.append(1e-7)
            
            past_type = current_type
            current_type = type
            past_start = current_start
            current_start = i
            past_end = i - 1
    
    energy_unvoiced = np.log(np.array(energy_unvoiced))
    
    cov = sklearn.covariance.EllipticEnvelope(random_state=0).fit(np.reshape(energy_unvoiced,(-1,1)))
    pred = cov.predict(np.reshape(energy_unvoiced,(-1,1)))
    
    # Gives start and end indexes for unvoiced sections
    unv_start = []
    unv_end = []
    
    voiced = False if class_resampled[0] == index_unvoiced else True
    unvoiced_start = 0
    unvoiced_end = 0
    count = 0
    print(np.count_nonzero(pred == -1), "out of", len(pred), "sections detected as 'unvoiced' removed (likely silent)")
    
    current_type = class_resampled[0]
    current_start = 0
    past_start = -1
    past_end = -1
    past_type = -1
    
    count = 0
    for i,type in enumerate(class_resampled):
        if type == current_type:
            continue
        else:
            if current_type == index_unvoiced:
                if pred[count] == 1:
                    unv_start.append(current_start)
                    unv_end.append(i)
                    
                count = count + 1
            
            past_type = current_type
            current_type = type
            past_start = current_start
            current_start = i
            past_end = i - 1
    
    return unv_start,unv_end

if __name__ == '__main__':
    # Parser settings
    parser = argparse.ArgumentParser(
        description='Parser',
        add_help=False
    )

    parser.add_argument(
        '--song',
        type=str,
        help='song path, wav mono or stereo'
    )
    
    parser.add_argument('--out-path',type=str,
                        default=None, help='Path for a txt file to be written')
    
    parser.add_argument('--outdir-wav',type=str,
                        default=None, help='Write to wav files in the given directory')
    
    args, _ = parser.parse_known_args()

    # if out-path not required, put it by default in the audio file's directory
    if args.out_path == None:
        args.out_path = os.path.dirname(args.song)
        
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    audio, rate = sf.read(
        args.song,
        always_2d=False
    ) # audio is a numpy array with size (nb_timesteps, nb_channels)
    
    print("Sampling rate:",rate,"Hz")
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    if len(audio.shape) == 1:
        audio_mono = audio
    
    elif audio.shape[1] == 2:
        audio_mono = audio.sum(axis=1) / 2
    
    unv_start,unv_end = detect(audio_mono,rate)
    
    output_file_name = os.path.splitext(ntpath.basename(args.song))[0] + '.csv'
    output_name = os.path.join(args.out_path,output_file_name)
    with open(output_name, 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(["FrameStart", "FrameEnd", "TimeStart", "TimeEnd"])
        rows = zip(unv_start,
                   unv_end,
                   [k/rate for k in unv_start],
                   [k/rate for k in unv_end]
                   )
        for row in rows:
            wr.writerow(row)

    
    if args.outdir_wav:
        if not os.path.exists(args.outdir_wav):
            os.makedirs(args.outdir_wav)
        
        for i in range(len(unv_start)):
            section_start = unv_start[i]/rate
            section_end = unv_end[i]/rate
            name = f"{section_start:.3f}" + '-' + f"{section_end:.3f}" + '.wav'
            sf.write(os.path.join(args.outdir_wav,name),
                    audio[unv_start[i]:unv_end[i]], 
                    samplerate=rate)