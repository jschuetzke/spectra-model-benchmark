
from tensorflow.keras import layers, optimizers, metrics
from tensorflow.keras.models import Model

def cnn(input_size=5000, dropout=.5, lr=1e-4):
    input_layer = layers.Input(shape=(input_size, 1), 
                               name="input")
    x = layers.Conv1D(64, 3, strides=1, padding='same',
                      activation='relu')(input_layer)
    x = layers.Conv1D(64, 3, strides=1, padding='same',
                      activation='relu')(x)
    x = layers.MaxPool1D(2, strides=2)(x)
    x = layers.Conv1D(128, 3, strides=1, padding='same',
                      activation='relu')(x)
    x = layers.Conv1D(128, 3, strides=1, padding='same',
                      activation='relu')(x)
    x = layers.MaxPool1D(2, strides=2)(x)
    x = layers.Conv1D(256, 3, strides=1, padding='same',
                      activation='relu')(x)
    x = layers.Conv1D(256, 3, strides=1, padding='same',
                      activation='relu')(x)
    x = layers.Conv1D(256, 3, strides=1, padding='same',
                      activation='relu')(x)
    x = layers.Conv1D(256, 3, strides=1, padding='same',
                      activation='relu')(x)
    x = layers.MaxPool1D(2, strides=2)(x)
    x = layers.Conv1D(512, 3, strides=1, padding='same',
                      activation='relu')(x)
    x = layers.Conv1D(512, 3, strides=1, padding='same',
                      activation='relu')(x)
    x = layers.Conv1D(512, 3, strides=1, padding='same',
                      activation='relu')(x)
    x = layers.Conv1D(512, 3, strides=1, padding='same',
                      activation='relu')(x)
    x = layers.MaxPool1D(2, strides=2)(x)
    x = layers.Conv1D(512, 3, strides=1, padding='same',
                      activation='relu')(x)
    x = layers.Conv1D(512, 3, strides=1, padding='same',
                      activation='relu')(x)
    x = layers.Conv1D(512, 3, strides=1, padding='same',
                      activation='relu')(x)
    x = layers.Conv1D(512, 3, strides=1, padding='same',
                      activation='relu')(x)
    x = layers.MaxPool1D(2, strides=2)(x)
    x = layers.Flatten(name='flat')(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)
    opt = optimizers.Adam(learning_rate=lr)
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
