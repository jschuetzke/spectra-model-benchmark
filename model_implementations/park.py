
from tensorflow.keras import layers, optimizers, metrics
from tensorflow.keras.models import Model
from .basic_blocks import get_dense_stack

def cnn(input_size=5000, dropout=.3, dense_neurons=[700, 70],
        lr=1e-4):
    input_layer = layers.Input(shape=(input_size, 1),
                               name="input")
    x = layers.Conv1D(80, 100, strides=5, padding='same',
                      activation='relu', 
                      name='conv1')(input_layer)
    x = layers.AvgPool1D(3, strides=2, 
                         name='maxpool1')(x)
    x = layers.Conv1D(80, 50, strides=5, padding='same',
                      activation='relu', 
                      name='conv2')(x)
    x = layers.AvgPool1D(3, strides=1, 
                         name='maxpool2')(x)
    x = layers.Conv1D(80, 25, strides=2, padding='same',
                      activation='relu', 
                      name='conv3')(x)
    x = layers.AvgPool1D(3, strides=1, 
                         name='maxpool3')(x)
    x = layers.Flatten(name='flat')(x)
    x = get_dense_stack(x, dense_neurons, dropout, activation="relu")
    out = layers.Dense(1, activation='sigmoid', 
                       name='output')(x)
    model = Model(input_layer, out)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[
            metrics.BinaryAccuracy(name="acc"),
            metrics.Precision(name="prc"),
            metrics.Recall(name="rec"),
        ],
    )
    return model

