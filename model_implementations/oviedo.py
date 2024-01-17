
from tensorflow.keras import layers, optimizers, metrics
from tensorflow.keras.models import Model

def cnn(input_size=5000, lr=1e-4):
    input_layer = layers.Input(shape=(input_size, 1),
                               name="input")
    x = layers.Conv1D(32, 8, strides=8, padding='same',
                      activation='relu', 
                      name='conv1')(input_layer)
    x = layers.Conv1D(32, 5, strides=5, padding='same',
                      activation='relu', 
                      name='conv2')(x)
    x = layers.Conv1D(32, 3, strides=3, padding='same',
                      activation='relu', 
                      name='conv3')(x)
    x = layers.GlobalAveragePooling1D()(x)
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