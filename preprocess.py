import pickle as pk
import numpy as np
import mne as mne
from mne.decoding import CSP

# Read data from exact path
def read_data(path):
    with open(path,'rb') as fo:
        file = pk.load(fo,encoding='bytes')
    data=file['data']
    return data

# convert array into raw,event
def array2Raw(data,channel_map,freq):
    channel_Num=np.shape(data)[0]
    channel_names=[channel_map[i] for i in range(1,channel_Num+1)]
    ch_tp=['eeg' for _ in range(59)]
    ch_tp.extend(['ecg','eog','eog','eog','eog','stim'])
    info=mne.create_info(ch_names=channel_names,sfreq=freq,ch_types=ch_tp)
    raw=mne.io.RawArray(data,info)
#     add events
    events=mne.find_events(raw)
    return raw,events

# convert dataframe into X and Y
# def df2matrix(df,trial,channel,timepoint,event_map):
#     X=np.zeros((trial,channel,timepoint))
#     Y=np.zeros((trial,1))
#     df=df.values
#     row,column=np.shape(df)
#     epoch=1
#     index=0
#     t=0
#     while index<row:
#         if df[index][2]==epoch:
#             X[t,:,:]=np.transpose(df[index:index+timepoint,3:column])
#             Y[t,:]=event_map[df[index][1]]
#             t+=1
#             index+=timepoint
#         epoch+=1
#     return X,Y

def convert2XY(data):
    trail=np.shape(data)[0]
    Y = np.zeros((trail, 1))
    for i in range(trail):
        Y[i,:]=data[i,59,0]
    return data[:,:59,:],Y

# preprocess
def preprocess_pipeline(path,channel_map,event_map,freq=250,l_feq=0,h_feq=30,trial=500,channel=59,timepoint=250):
    # read data and convert it to raw and event
    data=read_data(path)
    raw,events=array2Raw(data,channel_map,freq)
    picks=mne.pick_types(raw.info,eeg=True,stim=True)
    # bandpass filtering
    raw=raw.filter(l_freq=l_feq,h_freq=h_feq)
    # extract epochs with baseline and convert it to dataframe
    epochs=mne.Epochs(raw,events,event_id=event_map,tmin=0,tmax=0.996,baseline=(None,None),picks=picks)
    epochs_data=epochs.get_data()
    X,Y=convert2XY(epochs_data)
    return X,Y

# normalize by channel
def channel_normalize(subject_data):
    trial,channel,timepoint=np.shape(subject_data)
    norm_X=np.zeros((trial,channel,timepoint))
    for t in range(trial):
        cur_data=subject_data[t,:,:]
        channel_max=np.amax(cur_data,1)
        channel_min=np.min(cur_data,1)
        channel_mean=np.mean(cur_data,1)
        for i in range(channel):
            norm_X[t,i,:]=(cur_data[i]-channel_mean[i])/(channel_max[i]-channel_min[1])
    return norm_X

# zip all trial from one subject
def train_run(id):
    trial_Num=21*500
    channel_Num=59
    timepoint_Num=250
    subject_data=np.zeros((trial_Num,channel_Num,timepoint_Num))
    subject_label=np.zeros((trial_Num,1))
    # original path
    path = r'D:\ERP\ERP\数据\有训练集ERP\A榜数据集\traindata\S0' + str(id)
    # path = r'../../dataset/ERP_B/traindata/S0' + str(id)
    index=0
    for b in range(1,22):
        if b<10:
            block=str(0)+str(b)
        else:
            block=str(b)
        temp_path = path + r'/block' + block + '.pkl'
        channel_map = {1: 'Fpz', 2: 'Fp1', 3: 'Fp2', 4: 'AF3', 5: 'AF4', 6: 'AF7', 7: 'AF8', 8: 'FZ',
                       9: 'F1', 10: 'F2', 11: 'F3', 12: 'F4', 13: 'F5', 14: 'F6', 15: 'F7', 16: 'F8',
                       17: 'FCz', 18: 'FC1', 19: 'FC2', 20: 'FC3', 21: 'FC4', 22: 'FC5', 23: 'FC6', 24: 'FT7',
                       25: 'FT8', 26: 'Cz', 27: 'C1', 28: 'C2', 29: 'C3', 30: 'C4', 31: 'C5', 32: 'C6',
                       33: 'T7', 34: 'T8', 35: 'CP1', 36: 'CP2', 37: 'CP3', 38: 'CP4', 39: 'CP5', 40: 'CP6',
                       41: 'TP7', 42: 'TP8', 43: 'Pz', 44: 'P3', 45: 'P4', 46: 'P5', 47: 'P6', 48: 'P7',
                       49: 'P8', 50: 'POz', 51: 'PO3', 52: 'PO4', 53: 'PO5', 54: 'PO6', 55: 'PO7', 56: 'PO8',
                       57: 'Oz', 58: 'O1', 59: 'O2', 60: 'ECG', 61: 'HEOR', 62: 'HEOL', 63: 'VEOU', 64: 'VEOL',
                       65: 'event'}
        event_map={'non-target':1,'car':3,'people':2}
        block_data, block_label = preprocess_pipeline(temp_path,channel_map,event_map)
        subject_data[500 * index:500 * (index + 1)] = block_data
        subject_label[500 * index:500 * (index + 1)] = block_label
        index += 1
    subject_norm_data=channel_normalize(subject_data)
    return subject_norm_data,subject_label

def test_run(id):
    trial_Num=9*500
    channel_Num=59
    timepoint_Num=250
    subject_data=np.zeros((trial_Num,channel_Num,timepoint_Num))
    subject_label=np.zeros((trial_Num,1))
    # original path
    # path = r'../../dataset/ERP_B/testdata/S0' + str(id)
    path = r'D:\ERP\ERP\数据\有训练集ERP\B榜数据集\testdata\S0' + str(id)
    index=0
    for b in range(1,10):
        block=str(0)+str(b)
        temp_path = path + r'/block' + block + '.pkl'
        channel_map = {1: 'Fpz', 2: 'Fp1', 3: 'Fp2', 4: 'AF3', 5: 'AF4', 6: 'AF7', 7: 'AF8', 8: 'FZ',
                       9: 'F1', 10: 'F2', 11: 'F3', 12: 'F4', 13: 'F5', 14: 'F6', 15: 'F7', 16: 'F8',
                       17: 'FCz', 18: 'FC1', 19: 'FC2', 20: 'FC3', 21: 'FC4', 22: 'FC5', 23: 'FC6', 24: 'FT7',
                       25: 'FT8', 26: 'Cz', 27: 'C1', 28: 'C2', 29: 'C3', 30: 'C4', 31: 'C5', 32: 'C6',
                       33: 'T7', 34: 'T8', 35: 'CP1', 36: 'CP2', 37: 'CP3', 38: 'CP4', 39: 'CP5', 40: 'CP6',
                       41: 'TP7', 42: 'TP8', 43: 'Pz', 44: 'P3', 45: 'P4', 46: 'P5', 47: 'P6', 48: 'P7',
                       49: 'P8', 50: 'POz', 51: 'PO3', 52: 'PO4', 53: 'PO5', 54: 'PO6', 55: 'PO7', 56: 'PO8',
                       57: 'Oz', 58: 'O1', 59: 'O2', 60: 'ECG', 61: 'HEOR', 62: 'HEOL', 63: 'VEOU', 64: 'VEOL',
                       65: 'event'}
        event_map={'non-target':1,'car':3,'people':2}
        block_data, block_label = preprocess_pipeline(temp_path,channel_map,event_map)
        subject_data[500 * index:500 * (index + 1)] = block_data
        subject_label[500 * index:500 * (index + 1)] = block_label
        index += 1
    subject_norm_data=channel_normalize(subject_data)
    return subject_norm_data,subject_label

def split_data(X,Y,rate):
    label=Y.flatten()
    total_trial=np.shape(X)[0]
    # account of train trial of all trials
    train_trial=int(total_trial*rate)

#     split train set and test set
    train_X=X[:train_trial,:,:]
    train_Y=Y[:train_trial]
    test_X=X[train_trial:,:,:]
    test_Y=Y[train_trial:]

    return train_X,test_X,train_Y,test_Y

def pick_data(X,Y,oth_no=1,tar_no=2):
    label=Y.flatten()
    oth_id=label==oth_no
    tar_id=label==tar_no

    oth_sig=X[oth_id,:,:]
    oth_label=Y[oth_id]
    tar_sig=X[tar_id,:,:]
    tar_label=Y[tar_id]

    sig_12=np.concatenate([oth_sig,tar_sig],axis=0)
    label_12=np.concatenate([oth_label,tar_label],axis=0)
    return sig_12,label_12
