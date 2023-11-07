import tensorflow as tf
import os, math
import numpy as np
import tensorflow.keras.backend as K


if not os.path.exists('results'):
    os.mkdir('results')
if not os.path.exists('models'):
    os.mkdir('models')


def lr_schedule(epoch, lr):
    initial_lr = 1e-3
    decay_factor = 0.5
    decay_step = 5
    if epoch > 50:
        epoch = 50
    lr = initial_lr * (decay_factor ** (epoch // decay_step))
    return lr
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)


class GaSNet3:
    """
    Initialize, setting the input pixel, strat wavelength, end wavelength, and output channel
    and the network name
    """
    def __init__(self,Network_name, labels, task='classification', scale_factor=1, pixel_number=3600):
        self.Network_name = Network_name
        self.task = task
        self.Input_pixel = pixel_number
        self.Inpt = tf.keras.layers.Input(shape=(self.Input_pixel,1)) #shape of spectra
        self.batch = 128 # training batch
        self.class_names = labels
        self.path = 'models/'+Network_name
        self.regression_range = [0,5]
        self.scale_factor = scale_factor
        self.initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1e-3)
        
        
        if self.task=='classification':
            self.MC_num = 1 #Monte-Carlo times
            self.Output_channel = len(labels)
        elif self.task=='regression':
            self.MC_num = 10 #Monte-Carlo times
            self.Output_channel = 1

        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def Block_ResNet(self, x0, n, kernel=5, stride=3, short_cut=True):
        """
        one ResNet Block, to reduce feature dimension
        """
        x = tf.keras.layers.Conv1D(n, kernel_size=kernel,strides=1,padding='same',kernel_initializer=self.initializer)(x0)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv1D(n, kernel_size=kernel,strides=stride, kernel_initializer=self.initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=kernel,strides=1,padding='same')(x)
        if short_cut:
            ShortCut = tf.keras.layers.Conv1D(n, kernel_size=kernel, strides=stride, kernel_initializer=self.initializer)(x0)
        x = tf.keras.layers.Add()([x,ShortCut])
        return x
    
    def ResNet_subNet(self,inpts,activation='linear'):
        """
        Networks model made by Blocks
        """
        x = self.Block_ResNet(inpts,16)
        x = self.Block_ResNet(x,32)
        x = self.Block_ResNet(x,64)
        x = self.Block_ResNet(x,128)
        x = self.Block_ResNet(x,256)
        x = self.Block_ResNet(x,512)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=self.initializer)(x)
        outputs = tf.keras.layers.Dense(self.Output_channel,activation=activation, kernel_initializer=self.initializer)(x)
        return outputs
    
    # Monte-Carlo the sub-network, MC_num times.
    def ResNet_MC(self,inpts,activation='linear'):
        if self.MC_num > 1:
            outputs = tf.keras.layers.concatenate([self.ResNet_subNet(inpts,activation) for i in range(self.MC_num)],axis=-1) # MC_num times
        else:
            outputs = self.ResNet_subNet(inpts,activation)
        model = tf.keras.Model(inpts, outputs, name=self.Network_name)
        return model

    
    def Plot_Model(self):
          """
          Plot the network architecture
          """
          model = self.ResNet_MC(self.Inpt)
          model.summary()
          tf.keras.utils.plot_model(model,to_file=self.path+'/'+model.name+'.pdf',show_shapes=True,show_layer_names=True)

    def label_preprocess(self,label):
        """
        reshape the label and reshift array, conevrt the label to one-hot code.
        """
        if self.task == 'classification':
            label = np.array(label)
            value = np.vectorize(self.class_names.get)(label)
            label = tf.keras.utils.to_categorical(value,num_classes=self.Output_channel)
            return label

        elif self.task == 'regression':
            label = np.array(label)
            label = label.reshape(len(label),1)
            label = np.clip(label, self.regression_range[0], self.regression_range[1])
            label = label*self.scale_factor
            label = tf.convert_to_tensor(label)
            label = tf.keras.layers.concatenate([label for i in range(self.MC_num)],axis=-1) # MC_num times
            return label
        
    def data_preprocess(self,flux):
        """
        The input flux and label should be propocess
        """ 
        flux = tf.keras.utils.normalize(flux,axis=-1)
        flux = flux * self.Input_pixel **(1/2)
        return flux
    

    def Train_Model(self,data,lr=1e-3,epo=50):
        """
        Training the model.
        Input training data.
        """
        if self.task == 'classification':
            activate = 'softmax'
            metric = 'acc'
            monitor = 'val_acc'
            loss = 'categorical_crossentropy'

        elif self.task == 'regression':
            activate = 'linear'
            metric = 'mae'
            monitor = 'val_mae'
            loss = tf.keras.losses.Huber(0.1)

        train_x, train_y = self.data_preprocess(data['train']['flux']),self.label_preprocess(data['train']['label'])
        valid_x, valid_y = self.data_preprocess(data['valid']['flux']),self.label_preprocess(data['valid']['label'])

        model = self.ResNet_MC(self.Inpt, activation=activate) #need to creat network first when you training
        optimizer = tf.keras.optimizers.Adam(learning_rate = lr) # Adam 
        model.compile(optimizer, loss=loss, metrics=metric)  # complize model
        if os.path.exists(self.path+'/'+self.Network_name+'.h5'):
              model.load_weights(self.path+'/'+self.Network_name+'.h5')
              print('loading the existed model')
        # https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks/ModelCheckpoint
        checkPoint = tf.keras.callbacks.ModelCheckpoint(self.path+'/'+model.name+'.h5',monitor=monitor,verbose=1,save_best_only=True,save_weights_only=False)# callback function
        csvLogger = tf.keras.callbacks.CSVLogger(self.path+'/'+model.name+'.csv',append=True) # save training history
        leran_rate_schedule = lr_scheduler #warmup cosin learning rate
        model.fit(train_x,train_y,epochs=epo,batch_size=self.batch,validation_data=(valid_x,valid_y),callbacks=[checkPoint,csvLogger,leran_rate_schedule],shuffle=True)

    def Prodiction(self,flux):
        """
        predict the classes or redshift.
        """ 
        model = tf.keras.models.load_model(self.path+'/'+self.Network_name+'.h5')
        flux = self.data_preprocess(flux)
        pred = model.predict(flux) # give classes and reshift

        if self.task == 'classification':
            pred_label = np.argmax(pred,axis=-1) # turn one-hot to integer value, get the max value index
            dict = {v:k for k, v in self.class_names.items()} #  reverse key and value of a dict
            pred_label = np.vectorize(dict.get)(pred_label) # turn integer value to its name
            return pred_label

        elif self.task == 'regression':
            pred = pred / self.scale_factor
            pred_hat = np.mean(pred,axis=-1)
            pred_std = np.std(pred,axis=-1)
            return pred_hat, pred_std