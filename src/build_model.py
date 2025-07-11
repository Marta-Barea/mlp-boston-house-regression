import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def build_model(input_dim, units=13, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(units, input_dim=input_dim,
              activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='linear', kernel_initializer='glorot_uniform'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    return model
