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
def compute_class_weight(class_weight, *, classes, y):
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
