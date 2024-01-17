import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import model_implementations as mi
import wandb
from wandb.keras import WandbCallback, WandbMetricsLogger

MODEL_LIST = [
    "cnn2",
    "cnn3",
    "inc3",
    "inc6",
    "cnn6",
    "cnn_bn",
    "vgg",
    "resnet",
    "park",
    "vecsei",
    "fan",
    "oviedo",
    "sang",
    "mozaffari",
    "bhattacharya",
    "schuetzke",
]


def test_model(model_name):
    # import data
    xt = np.load("x_train.npy")
    xt /= np.max(xt, axis=1, keepdims=True)
    xv = np.load("x_val.npy")
    xv /= np.max(xv, axis=1, keepdims=True)

    yt = np.load("y_train.npy")
    yv = np.load("y_val.npy")

    wandb.init(project="spectra-benchmark", name=model_name, reinit=True)

    histories = []
    precisions = np.zeros([5])

    for i in range(5):
        model = None
        model = getattr(mi, model_name)(input_size=xt.shape[1])
        hist = model.fit(
            xt,
            yt,
            128,
            epochs=1000,
            verbose=2,
            callbacks=[
                EarlyStopping(patience=20, restore_best_weights=True),
                ReduceLROnPlateau(patience=15, factor=0.5),
            ],
            validation_data=(xv, yv),
        )
        logs = model.evaluate(xv, yv)
        precisions[i] = logs[2]
        histories.append(hist)
    print(precisions.round(4))
    ix = np.argsort(precisions)[2]
    h = histories[ix]
    epochs = len(h.history["loss"])
    for i in range(epochs):
        log_epoch = {
            "loss": h.history["loss"][i],
            "acc": h.history["acc"][i],
            "prc": h.history["prc"][i],
            "rec": h.history["rec"][i],
            "val_loss": h.history["val_loss"][i],
            "val_acc": h.history["val_acc"][i],
            "val_prc": h.history["val_prc"][i],
            "val_rec": h.history["val_rec"][i],
            "lr": h.history["lr"][i],
        }
        wandb.log(log_epoch)
    wandb.log({"best_prc": precisions[ix], "param_count": model.count_params()})
    wandb.finish(quiet=True)
    return


def main():
    for m in MODEL_LIST:
        test_model(m)
    return


if __name__ == "__main__":
    main()
