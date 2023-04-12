from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import tensorflow_addons as tfa
import scipy.io as scio
# update
from utils import compute_class_weight, total_evaluate
from tensorflow.keras.constraints import max_norm
# update
import preprocess as pp
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

def normalize_matrix(matrix):
    minVals=np.min(matrix,axis=0)
    maxVals=np.max(matrix,axis=0)
    range=maxVals-minVals
    m=np.shape(matrix)[0]
    diffnormData=matrix-np.tile(minVals,(m,1))
    normData=diffnormData/np.tile(range,(m,1))
    return normData

def cal_DAU(data,max_iter=100,step_size=0.01):
    trial,channel,timepoint=np.shape(data)

    # initial A,D,U
    A=np.mean(data,axis=0)
    D=np.ones((trial,channel,channel))
    U=np.ones((trial,timepoint,timepoint))

    # gradient descent
    for iter in range(max_iter):
        delta_A =np.zeros((channel,timepoint))
        for cur in range(trial):
            # get current D,U,X
            cur_X=data[cur,:,:]
            cur_D=D[cur,:,:]
            cur_U=U[cur,:,:]
            # calculate the common part:DAU-X
            diff=cur_D@A@cur_U-cur_X

            # calculate the delta_D,delta_U
            delta_D=2*diff@cur_U.T@A.T
            delta_U=2*A.T@cur_D.T@diff

            # normalize delta matrix
            nor_delta_D=normalize_matrix(delta_D)
            nor_delta_U=normalize_matrix(delta_U)

            # update Di,Ui
            D[cur,:,:]=cur_D-step_size*nor_delta_D
            U[cur,:,:]=cur_U-step_size*nor_delta_U

            # calculate cur_delta_A
            diff2=D[cur,:,:]@A@U[cur,:,:]-cur_X
            cur_delta_A=2*D[cur,:,:].T@diff2@U[cur,:,:].T
            nor_cur_delta_A=normalize_matrix(cur_delta_A)

            # calculate delta_A sum
            delta_A = delta_A + nor_cur_delta_A

        #     update A
        nor_delta_A=normalize_matrix(delta_A)
        A=A-step_size*nor_delta_A
    D_mean=np.mean(D,axis=0)
    U_mean=np.mean(U,axis=0)

    return D_mean,A,U_mean


def get_data_in_single_label(data,label,get_no):
    label=label.flatten()
    id=(label==get_no)
    get_data=data[id,:,:]
    return get_data



def get_A(data,label,tar1_no=1,tar2_no=2,max_iter=100,step_size=0.01):
    label = label.flatten()
    tar1_id = (label == tar1_no)
    tar2_id = (label == tar2_no)

    tar1_data=data[tar1_id,:,:]
    tar2_data=data[tar2_id,:,:]

    D_tar1,A_tar1,U_tar1=cal_DAU(tar1_data,max_iter,step_size)
    D_tar2,A_tar2,U_tar2=cal_DAU(tar2_data,max_iter,step_size)

    return A_tar1,A_tar2


def mean_squared_error(y_true, y_pred):
    """ loss function computing MSE of non-blank(!=0) in y_true
		Args:
			y_true(tftensor): true label
			y_pred(tftensor): predicted label
		return:
			MSE reconstruction error for loss computing
	"""
    loss = K.switch(K.equal(y_true, tf.constant(0.)), tf.zeros(K.shape(y_true)), K.square(y_pred - y_true))
    return K.mean(loss, axis=-1)

def triplet_loss(margin = 1.0):
    def inner_triplet_loss_objective(y_true, y_pred):
        labels = y_true
        embeddings = y_pred
        loss=tfa.losses.triplet_semihard_loss(y_true=labels, y_pred=embeddings,margin=margin)
        return loss
    return inner_triplet_loss_objective

def SparseCategoricalCrossentropy(class_weight=None):
    """[SparseCategoricalCrossentropy]

    Args:
        class_weight ([dict], optional): dict of class_weight
        class_weight = {0: 0.3,
                        1: 0.7}
        Defaults to None.
    """

    def inner_sparse_categorical_crossentropy(y_true, y_pred):
        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        if class_weight:
            keys_tensor = tf.cast(tf.constant(list(class_weight.keys())), dtype=tf.int32)
            vals_tensor = tf.constant(list(class_weight.values()), tf.float32)
            input_tensor = tf.cast(y_true, dtype=tf.int32)
            init = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
            table = tf.lookup.StaticHashTable(init, default_value=-1)
            sample_weight = table.lookup(input_tensor)
        else:
            sample_weight = None
        return scce(y_true, y_pred, sample_weight)

    return inner_sparse_categorical_crossentropy


def split_window(X):
    trial=np.shape(X)[0]
    X_sliding=np.zeros((trial,59,125,6))
    for i in range(6):
        X_sliding[:,:,:,i]=X[:,:,i*25:i*25+125]
    return X_sliding

def new_net(Chans=59,Samples=125,Windows=6):
    # encoder input
    encoder_input=Input(shape=(Chans,Samples,Windows))
    # block1: DepthwiseConv2D+BN+Activation+Dropout; DepthwiseConv2D+BN+Activation
    encoder=DepthwiseConv2D((1,60),depth_multiplier=8,
                            input_shape=(Chans,Samples,Windows),use_bias=False,
                            depthwise_constraint=max_norm(1.))(encoder_input)
    encoder=BatchNormalization()(encoder)
    encoder=Activation('elu')(encoder)
    encoder=Dropout(0.5)(encoder)

    encoder=DepthwiseConv2D((1,31),depth_multiplier=1,
                            use_bias=False,depthwise_constraint=max_norm(1.))(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('elu')(encoder)

    # block2:DepthwiseConv2D+BN+AC+AP+Reshape+spatialDropout
    encoder=DepthwiseConv2D((59,1),depth_multiplier=1,
                            use_bias=False, depthwise_constraint=max_norm(1.))(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('elu')(encoder)
    encoder=AveragePooling2D((1,8),strides=(1,4))(encoder)
    encoder=Reshape((1,48,8))(encoder)
    encoder=SpatialDropout2D(0.5)(encoder)

    # block 3: SeparableConv2D+BN+AC+AP+dropout
    encoder=SeparableConv2D(16,(1,16),use_bias=False)(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('elu')(encoder)
    encoder=AveragePooling2D((1,12),strides=(1,3))(encoder)
    encoder=Dropout(0.5)(encoder)

    # block 4: Flatten+FC
    encoder=Flatten()(encoder)
    encoder_output=Dense(64,kernel_constraint=max_norm(0.5),name='encoder_output')(encoder)

    # encoder model
    encoder_model=Model(inputs=encoder_input,outputs=encoder_output,name='encoder_net')

    # decoder
    decoder_input=Input(shape=64,name='decoder_input')
    decoder=Dense(128,activation='elu',kernel_constraint=max_norm(0.5))(decoder_input)
    decoder=Reshape((1,8,16))(decoder)
    decoder=Conv2DTranspose(8,(1,13),strides=(1,5),activation='elu')(decoder)
    decoder=Conv2DTranspose(8,(59,1),activation='elu')(decoder)
    decoder=Reshape((59,64,6))(decoder)
    decoder_output=Conv2DTranspose(6,(1,62),activation='elu')(decoder)

    # decoder_model
    decoder_model=Model(inputs=decoder_input,outputs=decoder_output, name='decoder_net')

    latent=encoder_model(encoder_input)
    train_xr=decoder_model(latent)
    z=Dense(3,activation='softmax', kernel_constraint=max_norm(0.5), name='classifier')(latent)

    return Model(inputs=encoder_input,outputs=[train_xr,latent,z],name='model')


def reconstract_X(X,Y,A1,A2):
    trail=np.shape(X)[0]
    rx=np.zeros((trail,59,250))
    for i in range(trail):
        if Y[i]==0:
            rx[i,:,:]=X[i,:,:]
        elif Y[i]==1:
            rx[i,:,:]=A1
        else:
            rx[i,:,:]=A2
    return rx



tf.config.experimental.list_physical_devices('GPU')
cm_name='confusionM.csv'
cm=np.zeros((18,3))
for sub in range(1, 7):
    #read train and test data
    train_X, train_Y = pp.train_run(sub)
    train_Y = train_Y.flatten() - 1
    test_X, test_Y = pp.test_run(sub)
    test_Y = test_Y.flatten() - 1

    # reconstruct A_tar1 and A_tar2
    A_tar1, A_tar2 = get_A(train_X, train_Y)
    rtrain_X = reconstract_X(train_X, train_Y, A_tar1, A_tar2)

    # split time window
    rtrain_X = split_window(rtrain_X)
    train_X = split_window(train_X)
    test_X = split_window(test_X)

    class_weight = compute_class_weight(train_Y)
    model = new_net()
    path = 'model_sub' + str(sub) + '.h5'
    checkpointer = ModelCheckpoint(monitor='loss', filepath=path,
                                   verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5,
                                  factor=0.5, mode='min', verbose=1,
                                  min_lr=1e-3)
    es = EarlyStopping(monitor='loss', mode='min', verbose=1,
                       patience=20)

    model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                  loss=[mean_squared_error, triplet_loss(margin=1.0),
                        SparseCategoricalCrossentropy(class_weight=class_weight)],
                  loss_weights=[0.3,0.2,0.5])

    model.fit(x=train_X, y=[rtrain_X, train_Y,train_Y], batch_size=100, shuffle=False,
              epochs=200,
              callbacks=[checkpointer, reduce_lr, es])

    probs = model.predict(test_X)[2]
    predict = probs.argmax(axis=-1)
    confusionM = confusion_matrix(test_Y, predict)

    cm[(sub - 1) * 3:sub * 3, :] = confusionM
np.savetxt(cm_name, cm, delimiter=',')
