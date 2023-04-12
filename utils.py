import numpy as np
import csv
from scipy import signal
from scipy.signal import butter, filtfilt
import wget
import os
import time
import tensorflow as tf
from sklearn.utils import class_weight
from scipy.interpolate import CubicSpline 
from scipy import ndimage
import argparse

# lib path
PATH = os.path.dirname(os.path.realpath(__file__))

def load_raw(dataset):
    # folder_name = str(PATH)+'/datasets'
    folder_name = 'datasets'
    if dataset == 'OpenBMI':
        try:
            num_subjects = 54
            sessions = [1, 2]
            save_path = folder_name + '/' + dataset + '/raw'
            if save_path is not None:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            for session in sessions:
                for person in range(1, num_subjects+1):
                    file_name = '/sess{:02d}_subj{:02d}_EEG_MI.mat'.format(session,person)
                    if os.path.exists(save_path+file_name):
                        os.remove(save_path+file_name) # if exist, remove file
                    print('\n===Download is being processed on session: {} subject: {}==='.format(session, person))
                    url = 'ftp://parrot.genomics.cn/gigadb/pub/10.5524/100001_101000/100542/session{}/s{}{}'.format(session, person, file_name)
                    print('save to: '+save_path+file_name)
                    wget.download(url,  save_path+file_name)
            print('\nDone!')
        except:
            raise Exception('Path Error: file does not exist, please direccly download at http://gigadb.org/dataset/100542')
    elif dataset == 'BCIC2a':
        try:
            num_subjects = 9
            sessions = ['T', 'E']
            save_path = folder_name + '/' + dataset + '/raw'
            if save_path is not None:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

            for session in sessions:
                for person in range(1, num_subjects+1):
                    file_name = '/A{:02d}{}.mat'.format(person, session)
                    if os.path.exists(save_path+file_name):
                        os.remove(save_path+file_name) # if exist, remove file
                    print('\n===Download is being processed on session: {} subject: {}==='.format(session, person))
                    url = 'https://lampx.tugraz.at/~bci/database/001-2014'+file_name
                    print('save to: '+save_path+file_name)
                    wget.download(url, save_path+file_name)
            print('\nDone!')
        except:
            raise Exception('Path Error: file does not exist, please direccly download at http://bnci-horizon-2020.eu/database/data-sets')
    elif dataset == 'SMR_BCI':
        try:
            num_subjects = 14
            sessions = ['T', 'E']
            save_path = folder_name + '/' + dataset + '/raw'
            if save_path is not None:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            for session in sessions:
                for person in range(1, num_subjects+1):
                    file_name = '/S{:02d}{}.mat'.format(person, session)
                    if os.path.exists(save_path+file_name):
                        os.remove(save_path+file_name) # if exist, remove file
                    print('\n===Download is being processed on session: {} subject: {}==='.format(session, person))
                    url = 'https://lampx.tugraz.at/~bci/database/002-2014'+file_name
                    print('save to: '+save_path+file_name)
                    wget.download(url,  save_path+file_name)
            print('\nDone!')
        except:
            raise Exception('Path Error: file does not exist, please direccly download at http://bnci-horizon-2020.eu/database/data-sets')

class DataLoader:
    def __init__(self, dataset, train_type=None, data_type=None, num_class=2, subject=None, data_format=None, dataset_path='/datasets', **kwargs):

        self.dataset = dataset #Dataset name: 'OpenBMI', 'SMR_BCI', 'BCIC2a'
        self.train_type = train_type # 'subject_dependent', 'subject_independent'
        self.data_type = data_type # 'fbcsp', 'spectral_spatial', 'time_domain'
        self.dataset_path = dataset_path
        self.subject = subject # id, start at 1
        self.data_format = data_format # 'channels_first', 'channels_last'
        self.fold = None # fold, start at 1
        self.prefix_name = 'S'
        self.num_class = num_class
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])


        self.path = self.dataset_path+'/'+self.dataset+'/'+self.data_type+'/'+str(self.num_class)+'_class/'+self.train_type
    
    def _change_data_format(self, X):
        if self.data_format == 'NCTD':
            # (#n_trial, #channels, #time, #depth)
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        elif self.data_format == 'NDCT':
            # (#n_trial, #depth, #channels, #time)
            X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
        elif self.data_format == 'NTCD':
            # (#n_trial, #time, #channels, #depth)
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
            X = np.swapaxes(X, 1, 3)
        elif self.data_format == 'NSHWD':
            # (#n_trial, #Freqs, #height, #width, #depth)
            X = zero_padding(X)
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3], 1)
        elif self.data_format == None:
            pass
        else:
            raise Exception('Value Error: data_format requires None, \'NCTD\', \'NDCT\', \'NTCD\' or \'NSHWD\', found data_format={}'.format(self.data_format))
        print('change data_format to \'{}\', new dimention is {}'.format(self.data_format, X.shape))
        return X

    def load_train_set(self, fold, **kwargs):
        self.fold = fold
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])
    
        # load 
        X, y =  np.array([]),  np.array([])
        try:
            self.file_x = self.path+'/X_train_{}{:03d}_fold{:03d}.npy'.format(self.prefix_name, self.subject, self.fold)
            self.file_y = self.path+'/y_train_{}{:03d}_fold{:03d}.npy'.format(self.prefix_name, self.subject, self.fold)
            X = self._change_data_format(np.load(self.file_x))
            y = np.load(self.file_y)
        except:
            raise Exception('Path Error: file does not exist, please check this path {}, and {}'.format(self.file_x, self.file_y))
        return X, y

    def load_val_set(self, fold, **kwargs):
        self.fold = fold
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])

        # load 
        X, y =  np.array([]),  np.array([])
        try:
            self.file_x = self.path+'/X_val_{}{:03d}_fold{:03d}.npy'.format(self.prefix_name, self.subject, self.fold)
            self.file_y = self.path+'/y_val_{}{:03d}_fold{:03d}.npy'.format(self.prefix_name, self.subject, self.fold)
            X = self._change_data_format(np.load(self.file_x))
            y = np.load(self.file_y)
        except:
            raise Exception('Path Error: file does not exist, please check this path {}, and {}'.format(self.file_x, self.file_y))
        return X, y
    
    def load_test_set(self, fold, **kwargs):
        self.fold = fold
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])

        # load 
        X, y =  np.array([]),  np.array([])
        try:
            self.file_x = self.path+'/X_test_{}{:03d}_fold{:03d}.npy'.format(self.prefix_name, self.subject, self.fold)
            self.file_y = self.path+'/y_test_{}{:03d}_fold{:03d}.npy'.format(self.prefix_name, self.subject, self.fold)
            X = self._change_data_format(np.load(self.file_x))
            y = np.load(self.file_y)
        except:
            raise Exception('Path Error: file does not exist, please check this path {}, and {}'.format(self.file_x, self.file_y))
        return X, y


def compute_class_weight(class_weight, *, classes, y):
    """Estimate class weights for unbalanced datasets.

    Parameters
    ----------
    class_weight : dict, 'balanced' or None
        If 'balanced', class weights will be given by
        ``n_samples / (n_classes * np.bincount(y))``.
        If a dictionary is given, keys are classes and values
        are corresponding class weights.
        If None is given, the class weights will be uniform.

    classes : ndarray
        Array of the classes occurring in the data, as given by
        ``np.unique(y_org)`` with ``y_org`` the original class labels.

    y : array-like, shape (n_samples,)
        Array of original class labels per sample;

    Returns
    -------
    class_weight_vect : ndarray, shape (n_classes,)
        Array with class_weight_vect[i] the weight for i-th class

    References
    ----------
    The "balanced" heuristic is inspired by
    Logistic Regression in Rare Events Data, King, Zen, 2001.
    """
    # Import error caused by circular imports.
    from ..preprocessing import LabelEncoder

    if set(y) - set(classes):
        raise ValueError("classes should include all valid labels that can "
                         "be in y")
    if class_weight is None or len(class_weight) == 0:
        # uniform class weights
        weight = np.ones(classes.shape[0], dtype=np.float64, order='C')
    elif class_weight == 'balanced':
        # Find the weight of each class as present in y.
        le = LabelEncoder()
        y_ind = le.fit_transform(y)
        if not all(np.in1d(classes, le.classes_)):
            raise ValueError("classes should have valid labels that are in y")

        recip_freq = len(y) / (len(le.classes_) *
                               np.bincount(y_ind).astype(np.float64))
        weight = recip_freq[le.transform(classes)]
    else:
        # user-defined dictionary
        weight = np.ones(classes.shape[0], dtype=np.float64, order='C')
        if not isinstance(class_weight, dict):
            raise ValueError("class_weight must be dict, 'balanced', or None,"
                             " got: %r" % class_weight)
        for c in class_weight:
            i = np.searchsorted(classes, c)
            if i >= len(classes) or classes[i] != c:
                raise ValueError("Class label {} not present.".format(c))
            else:
                weight[i] = class_weight[c]

    return weight
def compute_class_weight(y_train):
    """compute class balancing

    Args:
        y_train (list, ndarray): [description]

    Returns:
        (dict): class weight balancing
    """
    return dict(zip(np.unique(y_train), 
                    class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(y_train),
                                                      y=y_train))) 
        
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self, save_path=None):
        self.save_path = save_path
    def on_train_begin(self, logs={}):
        self.logs = []
        if self.save_path:
            write_log(filepath=self.save_path, data=['time_log'], mode='w')
    def on_epoch_begin(self, epoch, logs={}):
        self.start_time = time.time()
    def on_epoch_end(self, epoch, logs={}):
        time_diff = time.time()-self.start_time
        self.logs.append(time_diff)
        if self.save_path:
            write_log(filepath=self.save_path, data=[time_diff], mode='a')

def write_log(filepath='test.log', data=[], mode='w'):
    '''
    filepath: path to save
    data: list of data
    mode: a = update data to file, w = write a new file
    '''
    try:
        with open(filepath, mode) as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(data)
    except IOError:
        raise Exception('I/O error')

def zero_padding(data, pad_size=4):
    if len(data.shape) != 4:
        raise Exception('Dimension is not match!, must have 4 dims')
    new_shape = int(data.shape[2]+(2*pad_size))
    data_pad = np.zeros((data.shape[0], data.shape[1], new_shape, new_shape))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_pad[i,j,:,:] = np.pad(data[i,j,:,:], [pad_size, pad_size], mode='constant')
    print(data_pad.shape)
    return data_pad 


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def resampling(data, new_smp_freq, data_len):
    if len(data.shape) != 3:
        raise Exception('Dimesion error', "--> please use three-dimensional input")
    new_smp_point = int(data_len*new_smp_freq)
    data_resampled = np.zeros((data.shape[0], data.shape[1], new_smp_point))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_resampled[i,j,:] = signal.resample(data[i,j,:], new_smp_point)
    return data_resampled

def psd_welch(data, smp_freq):
    if len(data.shape) != 3:
        raise Exception("Dimension Error, must have 3 dimension")
    n_samples,n_chs,n_points = data.shape
    data_psd = np.zeros((n_samples,n_chs,89))
    for i in range(n_samples):
        for j in range(n_chs):
            freq, power_den = signal.welch(data[i,j], smp_freq, nperseg=n_points)
            index = np.where((freq>=8) & (freq<=30))[0].tolist()
            # print("the length of---", len(index))
            data_psd[i,j] = power_den[index]
    return data_psd

from sklearn import metrics as sk_metrics

def pick_label(label,predict,oth_no,tar_no):
    oth_id=label==oth_no
    tar_id=label==tar_no

    oth_label=label[oth_id]
    oth_predict=predict[oth_id]
    tar_label=label[tar_id]
    tar_predict=predict[tar_id]

    reconstract_label=np.concatenate([oth_label,tar_label])
    reconstract_predict=np.concatenate([oth_predict,tar_predict])

    # 更改predict中其余id
    for i in range(np.shape(reconstract_predict)[0]):
        if reconstract_predict[i]!=oth_no and reconstract_predict[i]!=tar_no:
            reconstract_predict[i]=oth_no

    # 处理tar_no=1 & oth_no=0:
    if tar_no!=1 or oth_no!=0:
        for i in range(np.shape(reconstract_label)[0]):
            # deal with reconstract_label
            if reconstract_label[i]==oth_no:
                reconstract_label[i]=0
            if reconstract_label[i]==tar_no:
                reconstract_label[i]=1
            #  deal with reconstract_predict
            if reconstract_predict[i]==oth_no:
                reconstract_predict[i]=0
            if reconstract_predict[i]==tar_no:
                reconstract_predict[i]=1
    return reconstract_label,reconstract_predict

from sklearn.metrics import confusion_matrix
def total_evaluate(probs,label):
    predict=probs.argmax(axis=-1)

    # 0-1
    label_01,predict_01=pick_label(label,predict,oth_no=0,tar_no=1)
    precision_01=sk_metrics.precision_score(label_01,predict_01)
    recall_01=sk_metrics.recall_score(label_01,predict_01)
    F1_01= sk_metrics.f1_score(label_01,predict_01)
    auc_model_01 = tf.keras.metrics.AUC()
    auc_model_01.update_state(label_01, predict_01)
    auc_01 = auc_model_01.result().numpy()

    # 0-2
    label_02,predict_02=pick_label(label,predict,oth_no=0,tar_no=2)
    precision_02=sk_metrics.precision_score(label_02,predict_02)
    recall_02=sk_metrics.recall_score(label_02,predict_02)
    F1_02=sk_metrics.f1_score(label_02,predict_02)
    auc_model_02 = tf.keras.metrics.AUC()
    auc_model_02.update_state(label_02, predict_02)
    auc_02 = auc_model_02.result().numpy()

    precision=(precision_01+precision_02)/2.0
    recall=(recall_01+recall_02)/2.0
    F1=(F1_01+F1_02)/2.0
    auc=(auc_01+auc_02)/2.0

    acc=sk_metrics.accuracy_score(label,predict)

    return precision,recall,F1,auc,acc