
from tensorflow.keras import layers, optimizers, metrics
from tensorflow.keras.models import Model
from .basic_blocks import conv1d, get_dense_stack

def cnn(input_size=5000, dropout=.5, dense_neurons=[2048], 
        lr=1e-4):
    input_layer = layers.Input(shape=(input_size, 1),
                               name="input")
    # Batch Norm between conv and activation so we use custom conv1d func
    # instead of default keras Conv1D layer
    x = conv1d(input_layer, 16, 21, activation='leakyrelu', batch_norm=True)
    x = layers.MaxPool1D(2, strides=2, 
                         name='maxpool1')(x)
    x = conv1d(x, 32, 11, activation='leakyrelu', batch_norm=True)
    x = layers.MaxPool1D(2, strides=2, 
                         name='maxpool2')(x)
    x = conv1d(x, 64, 5, activation='leakyrelu', batch_norm=True)
    x = layers.MaxPool1D(2, strides=2, 
                         name='maxpool3')(x)
    x = layers.Flatten(name='flat')(x)
    x = get_dense_stack(x, dense_neurons, dropout, activation=None)
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