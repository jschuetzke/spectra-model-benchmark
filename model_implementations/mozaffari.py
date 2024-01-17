
from tensorflow.keras import layers, optimizers, metrics
from tensorflow.keras.models import Model

def cnn(input_size=5000, lr=1e-4):
    input_layer = layers.Input(shape=(input_size, 1),
                               name="input")
    x = layers.Conv1D(32, 3, strides=1, padding='same',
                      activation='relu', 
                      name='conv1')(input_layer)
    
    x = layers.Flatten(name='flat')(x)
    x = layers.Dense(64, activation="relu")(x)
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

