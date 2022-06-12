
from tensorflow.keras  import backend as K

# from pickletools import optimize

import tensorflow as tf

from tensorflow.keras import *

from tensorflow.keras.losses import *

from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *

from tensorflow.keras.models import load_model, save_model


def my_crossentropy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * K.binary_crossentropy(y_pred, y_true), axis=-1)

def mymask(y_true):
    return K.minimum(y_true+1., 1.)

def msse(y_true, y_pred):
    return K.mean(mymask(y_true) * K.square(K.sqrt(y_pred) - K.sqrt(y_true)), axis=-1)

def mycost(y_true, y_pred):
     return K.mean(mymask(y_true) * (10*K.square(K.square(K.sqrt(y_pred) - K.sqrt(y_true))) + K.square(K.sqrt(y_pred) - K.sqrt(y_true)) + 0.01*K.binary_crossentropy(y_pred, y_true)), axis=-1)

def my_accuracy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * K.equal(y_true, K.round(y_pred)), axis=-1)


class reset_state_after_batch(tf.keras.callbacks.Callback):
    reset_after = 1 # reset state after N batch.
    curr = 0
    def on_batch_end(self, batch, logs=None):
        self.curr += 1
        if(self.curr >= self.reset_after):
            self.curr = 0
            self.model.reset_states()
        pass


def train(x_train, y_train, vad_train, batch_size=64, epochs=8, model_name="model.h5"):

    input_feature_size = x_train.shape[-1]
    output_feature_size = y_train.shape[-1]
    timestamp_size = batch_size
    input = Input(shape=(1, input_feature_size), batch_size=timestamp_size)

   
    # voice activity detection
    x1_1 = LSTM(24, return_sequences=True, stateful=True, recurrent_dropout=0.2)(input)
    x1_1 = Dropout(0.3)(x1_1)
    x1_2 = LSTM(24, return_sequences=True, stateful=True, recurrent_dropout=0.2)(x1_1)
    x1_2 = Dropout(0.3)(x1_2)
    x1_3 = LSTM(24, return_sequences=True, stateful=True, recurrent_dropout=0.2)(x1_2)
    x1_3 = Dropout(0.3)(x1_3)
    x = Flatten()(x1_3)
    x = Dropout(0.3)(x)
    x = Dense(1)(x)
    vad_output = Activation("hard_sigmoid")(x)

    # we dont concate input with layer output, because the range different will cause quite many quantisation lost.
    x_in = LSTM(64, return_sequences=True, stateful=True, recurrent_dropout=0.3)(input)

    # Noise spectral estimation
    x2 = concatenate([x_in, x1_1, x1_2, x1_3], axis=-1)
    x2 = LSTM(48, return_sequences=True, stateful=True, recurrent_dropout=0.3)(x2)
    x2 = Dropout(0.3)(x2)

    #Spectral subtraction
    x3 = concatenate([x_in, x2, x1_2, x1_3], axis=-1)
    x3 = LSTM(96, return_sequences=True, stateful=True, recurrent_dropout=0.3)(x3)
    x3 = Dropout(0.3)(x3)
    x = Flatten()(x3)
    x = Dense(output_feature_size)(x)
    x = Activation("hard_sigmoid")(x)

    """
        Simplified RNNoise-Like model. 
    """


    model = Model(inputs=input, outputs=[x, vad_output])
    #model.compile("adam", loss=[mycost, my_crossentropy], loss_weights=[10, 0.5], metrics=[msse])  # RNNoise loss and cost
    model.compile(optimizer=Adam(lr=0.0001), loss='mean_squared_error', metrics=[msse])
    model.summary()

    history = model.fit(x_train, [y_train, vad_train],
                        batch_size=timestamp_size, epochs=epochs, verbose=2, shuffle=False, # shuffle must be false
                        callbacks=[reset_state_after_batch()])# validation_split=0.1)

    # free the session to avoid nesting naming while we load the best model after.
    save_model(model, model_name)
    del model
    tf.keras.backend.clear_session()
    return history

def train_gains(x_train, y_train, batch_size=64, epochs=10, model_name="model.h5"):
    



    input_feature_size = x_train.shape[-1]
    output_feature_size = y_train.shape[-1]
    timestamp_size = batch_size

    Layer_In=Sequential()
    Layer_In.add(Input(shape=(1, input_feature_size), batch_size=timestamp_size))
    Layer_In.add(LSTM(64, return_sequences=True, stateful=True, recurrent_dropout=0.3))

    Layer_1_1=Sequential()
    Layer_1_1.add(Input(shape=(1, input_feature_size), batch_size=timestamp_size))
    Layer_1_1.add(LSTM(24, return_sequences=True, stateful=True, recurrent_dropout=0.2))
    Layer_1_1.add(Dropout(0.3))
    

    Layer_1_2=Sequential()
    Layer_1_2.add(Input(shape=(1, input_feature_size), batch_size=timestamp_size))
    Layer_1_2.add(LSTM(24, return_sequences=True, stateful=True, recurrent_dropout=0.2))
    Layer_1_2.add(Dropout(0.3))
    Layer_1_2.add(LSTM(24, return_sequences=True, stateful=True, recurrent_dropout=0.2))
    Layer_1_2.add(Input(shape=(1, input_feature_size), batch_size=timestamp_size))

    
    # Concat_result_1_1=Concatenate(axis=-1)([Layer_In, Layer_1_1])

    # Concat_result_1_2=Concatenate(axis=-1)([Concat_result_1_1, Layer_1_2])
    
    
    First_Concat=Sequential()
    First_Concat.add(Layer_In)
    First_Concat.add(Layer_1_1)
    First_Concat.add(Layer_1_2)
    First_Concat.add(LSTM(48, return_sequences=True, stateful=True, recurrent_dropout=0.3))
    First_Concat.add(Dropout(0.3))

    # Concat_result_2=concatenate([Layer_In, First_Concat,Layer_1_2],axis=-1)

    Second_Concat=Sequential()
    Second_Concat.add(Layer_In)
    Second_Concat.add(First_Concat)
    Second_Concat.add(Layer_1_2)
    Second_Concat.add(LSTM(96, return_sequences=True, stateful=True, recurrent_dropout=0.3))
    Second_Concat.add(Dropout(0.3))
    Second_Concat.add(Flatten())
    Second_Concat.add(Dense(output_feature_size))
    Second_Concat.add(Activation("hard_sigmoid"))
    Second_Concat.add(BatchNormalization())
    


    # x3 = concatenate([x_in, x2, x1_2], axis=-1)
    # x3 = LSTM(96, return_sequences=True, stateful=True, recurrent_dropout=0.3)(x3)
    # x3 = Dropout(0.3)(x3)
    # x = Flatten()(x3)
    # x = Dense(output_feature_size)(x)
    # x=BatchNormalization()(x)
    # x = Activation("hard_sigmoid")(x),




    #model.compile("adam", loss=[mycost, my_crossentropy], loss_weights=[10, 0.5], metrics=[msse])  # RNNoise loss and cost
    Second_Concat.compile(loss='mean_squared_error',optimizer=Adam(lr=0.0001,clipnorm=1))
    

   
    history = Second_Concat.fit(x_train, y_train,
                        batch_size=timestamp_size, epochs=epochs, verbose=2, shuffle=False, # shuffle must be false
                        callbacks=[reset_state_after_batch()])# validation_split=0.1)

    Second_Concat.summary()

    # free the session to avoid nesting naming while we load the best model after.
    save_model(Second_Concat, model_name)
    del Second_Concat
    tf.keras.backend.clear_session()
    return history