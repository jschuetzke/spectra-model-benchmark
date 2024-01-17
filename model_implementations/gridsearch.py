from tensorflow.keras import layers, optimizers, metrics
from tensorflow.keras.models import Model
from .basic_blocks import get_dense_stack


def cnn(
    input_size=5000,
    num_layers=3,
    num_conv=2,
    filters=64,
    kernel=51,
    dropout=0.3,
    dense_neurons=[200],
    hidden_act="relu",
    lr=3e-4,
):
    input_layer = layers.Input(shape=(input_size, 1), name="input")
    x = input_layer
    for l in range(num_layers):
        for _ in range(num_conv):
            x = layers.Conv1D(
                filters, kernel, padding="same", activation="relu"
            )(x)
        x = layers.MaxPool1D(2, strides=2)(x)
    x = layers.Flatten(name="flat")(x)
    x = get_dense_stack(x, dense_neurons, dropout, activation=hidden_act)
    opt = optimizers.Adam(learning_rate=lr)
    out = layers.Dense(1, activation="softmax", name="output")(x)
    model = Model(input_layer, out)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[metrics.BinaryAccuracy(name="accuracy")],
    )
    return model
