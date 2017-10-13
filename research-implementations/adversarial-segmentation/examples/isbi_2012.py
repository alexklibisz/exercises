import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
import tifffile as tiff

from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping
from keras.losses import binary_crossentropy as bce
from keras.models import Model, model_from_json
from keras import backend as K

import sys
sys.path.append('.')
from networks import UNet, ConvNetClassifier, set_trainable, \
    get_trainable_count, get_flat_weights, F1

np.random.seed(865)


def sampler(imgs, msks, input_shape, batch):
    """Generator that yields training batches randomly sampled 
    from given data with simple augmentations."""
    N, H, W = imgs.shape[:-1]
    h, w = input_shape[1:-1]
    ii = np.arange(N)
    aug_funcs = [
        lambda x: x,
        lambda x: np.rot90(x, 1, (0, 1)),
        lambda x: np.rot90(x, 2, (0, 1)),
        lambda x: np.rot90(x, 3, (0, 1)),
    ]
    while True:
        bii = np.random.choice(ii, batch, replace=(N <= batch))
        imgs_batch = np.zeros((batch, *input_shape[1:]), dtype=np.float32)
        msks_batch = np.zeros((batch, *input_shape[1:]), dtype=np.uint8)
        for i in range(batch):
            af = np.random.choice(aug_funcs)
            y0 = np.random.randint(0, H - h)
            x0 = np.random.randint(0, W - w)
            imgs_batch[i] = af(imgs[bii[i], y0:y0 + h, x0:x0 + w, :])
            msks_batch[i] = af(msks[bii[i], y0:y0 + h, x0:x0 + w, :])
        yield imgs_batch, msks_batch


def train_standard(S, imgs_trn, msks_trn, imgs_val, msks_val, steps_trn, steps_val, input_shape, epochs, batch):
    """Builds and trains the segmentation network by itself."""

    # Define sample generators. Generate a single validation set.
    gen_trn = sampler(imgs_trn, msks_trn, input_shape, batch)
    gen_val = sampler(imgs_val, msks_val, input_shape, batch * steps_val)
    x_val, y_val = next(gen_val)

    cb = [
        ModelCheckpoint('checkpoints/std_S_{val_loss:.2f}.hdf5',
                        monitor='val_loss', save_best_only=1, verbose=1, mode='min'),
        CSVLogger('checkpoints/std_history.csv')
    ]

    S.compile(optimizer=RMSprop(0.001, decay=1e-3), loss=bce)
    S.fit_generator(gen_trn, epochs=epochs, steps_per_epoch=steps_trn,
                    validation_data=(x_val, y_val), callbacks=cb)


def train_adversarial(S, D, imgs_trn, msks_trn, imgs_val, msks_val, iters_trn,
                      iters_val, epochs, batch, alpha0, alpha1, alpha_switch_epoch):
    """Builds and trains the segmentation network and adversarial classifier."""

    # Compile standalone segmentation network S.
    S.compile(optimizer=RMSprop(1e-3, decay=1e-3), loss=bce, metrics=[F1])

    # Custom metrics for monitoring D's outputs.
    def ytm(yt, yp):
        return K.mean(yt)

    def ypm(yt, yp):
        return K.mean(yp)

    # Compile standalone discriminator network D.
    D.compile(optimizer=Adam(0.001), loss=bce, metrics=['accuracy', ytm, ypm])

    # Combine segmentation (copy) with frozen discriminator to make generator.
    # Starts with a low alpha, later increased, based on Xue et. al.
    GS = model_from_json(S.to_json())
    GD = model_from_json(D.to_json())
    set_trainable(GD, False)
    G = Model(GS.input, outputs=[GS.output, GD(GS.output)])
    opt = RMSprop(1e-3, decay=1e-3)
    G.compile(optimizer=opt, loss=bce, loss_weights=[1, alpha0],
              metrics={'seg': [F1], 'model_2': ['accuracy']})

    # Check network instantiations.
    assert id(S) != id(GS)
    assert id(G.layers[-1]) != id(D)
    assert id(G.layers[-1]) == id(GD)
    assert get_trainable_count(G) == get_trainable_count(S)

    # Define sample generators and store a single validation set.
    gen_trn = sampler(imgs_trn, msks_trn, S.input_shape, batch * iters_trn)
    gen_val = sampler(imgs_val, msks_val, S.input_shape, batch * iters_val)
    imgs_epoch_val, msks_epoch_val = next(gen_val)

    # Pre-compute labels for generator and discriminator.
    gyt = np.ones((batch * iters_trn)) * 1.0    # Generator y-train.
    gyv = np.ones((batch * iters_val)) * 1.0    # Generator y-val.
    dytf = np.ones((batch * iters_trn)) * 0.0   # Discriminator y-train-fake.
    dytr = np.ones((batch * iters_trn)) * 1.0   # Discriminator y-train-real.
    dyvf = np.ones((batch * iters_val)) * 0.0   # Discriminator y-val-fake.
    dyvr = np.ones((batch * iters_val)) * 1.0   # Discriminator y-val-real.

    # Training loop. Training S, G, D in lock step.
    for e in range(epochs):

        # Update alpha for generator if switch point reached.
        if e == alpha_switch_epoch:
            print('Increasing alpha %.1lf -> %.1lf' % (alpha0, alpha1))
            opt = G.optimizer
            G.compile(optimizer=opt, loss=bce, loss_weights=[1, alpha1],
                      metrics={'seg': [F1], 'model_2': ['accuracy']})

        # New training set for this epoch.
        imgs_epoch_trn, msks_epoch_trn = next(gen_trn)

        # Fit segmentation model for one epoch.
        print('%03d: Fitting segmentation-only model' % e)
        xt, yt = imgs_epoch_trn, msks_epoch_trn
        xv, yv = imgs_epoch_val, msks_epoch_val
        S_history = S.fit(xt, yt, validation_data=(xv, yv), epochs=e + 1,
                          initial_epoch=e, batch_size=batch)

        # Update discriminator weights and fit generator for one epoch.
        print('%03d: Fitting adversarial segmentation model' % e)
        G.layers[-1].set_weights(D.get_weights())
        xt, yt = imgs_epoch_trn, [msks_epoch_trn, gyt]
        xv, yv = imgs_epoch_val, [msks_epoch_val, gyv]
        G_history = G.fit(xt, yt, validation_data=(xv, yv), epochs=e + 1,
                          initial_epoch=e, batch_size=batch)

        # Generate and combine fake and real data for the discriminator.
        xtf = GS.predict(imgs_epoch_trn, batch_size=batch)
        xvf = GS.predict(imgs_epoch_val, batch_size=batch)
        xtr = msks_epoch_trn
        xvr = msks_epoch_val
        xt, yt = np.concatenate([xtf, xtr]), np.concatenate([dytf, dytr])
        xv, yv = np.concatenate([xvf, xvr]), np.concatenate([dyvf, dyvr])

        # Train discriminator.
        print('%03d: Fitting adversarial discriminator model' % e)
        D_history = D.fit(xt, yt, validation_data=(xv, yv), epochs=e + 1,
                          initial_epoch=e, batch_size=batch)

        # Plot samples.
        S_msks_pred = S.predict(imgs_epoch_val[:10, ...], batch_size=batch)
        fig, _ = plt.subplots(10, 4, figsize=(8, 20))
        plt.suptitle('Epoch %d' % e)
        fig.axes[0].set_title('Image')
        fig.axes[1].set_title('True Mask')
        fig.axes[2].set_title('Adv. Mask')
        fig.axes[3].set_title('Std. Mask')
        for ax in fig.axes:
            ax.axis('off')
        for i in range(0, 10):
            fig.axes[i * 4 + 0].imshow(imgs_epoch_val[i, :, :, 0], cmap='gray')
            fig.axes[i * 4 + 1].imshow(xvr[i, :, :, 0], cmap='gray')
            fig.axes[i * 4 + 2].imshow(xvf[i, :, :, 0], cmap='gray')
            fig.axes[i * 4 + 3].imshow(S_msks_pred[i, :, :, 0], cmap='gray')
        plt.savefig('checkpoints/adv_sample_%02d.png' % e, dpi=150,
                    bbox_inches='tight', pad_inches=0)

        # Extract and save metrics.
        row = {
            'S F1 t': S_history.history['F1'][0],
            'S F1 v': S_history.history['val_F1'][0],
            'D Acc t': D_history.history['acc'][0],
            'D Acc v': D_history.history['val_acc'][0],
            'G-S F1 t': G_history.history['seg_F1'][0],
            'G-S F1 v': G_history.history['val_seg_F1'][0],
            'G-D Acc t': G_history.history['model_2_acc'][0],
            'G-D Acc v': G_history.history['val_model_2_acc'][0],
        }

        if e == 0:
            metrics = pd.DataFrame([row], columns=row.keys())
        else:
            metrics = metrics.append([row], ignore_index=True)
        metrics.to_csv('checkpoints/adv_metrics.csv', index=False)


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('--train', action='store_true', default=True)
    ap.add_argument('--adversarial', action='store_true', default=False)
    ap.add_argument('--submit', action='store_true', default=False)
    ap.add_argument('--model', type=str)
    ap.add_argument('--data_dir', type=str, default='data/isbi_2012')
    args = vars(ap.parse_args())

    # Paths for serializing model, reading data, etc.
    trn_imgs_path = 'data/isbi_2012/train-volume.tif'
    trn_msks_path = 'data/isbi_2012/train-labels.tif'
    tst_imgs_path = 'data/isbi_2012/test-volume.tif'
    tst_msks_path = 'data/isbi_2012/test-labels.tif'

    if args['train']:

        # Load data, images and labels have range [0, 255].
        imgs = tiff.imread(trn_imgs_path)[:, :, :, np.newaxis] / 255.
        msks = tiff.imread(trn_msks_path)[:, :, :, np.newaxis] / 255.

        # Normalize images.
        imgs -= np.mean(imgs)
        imgs /= np.std(imgs)

        # Train/val split.
        imgs_trn, msks_trn = imgs[:20, ...], msks[:20, ...]
        imgs_val, msks_val = imgs[20:, ...], msks[20:, ...]
        data = (imgs_trn, msks_trn, imgs_val, msks_val)

        # Network and training parameters.
        input_shape = (128, 128, 1)
        total_iters = 10000
        iters_trn = 500
        iters_val = 100
        batch = 8
        epochs = total_iters // iters_trn
        alpha0 = 0.1
        alpha1 = 1
        alpha_switch_epoch = 3

        # Networks.
        S = UNet(input_shape)
        D = ConvNetClassifier(S.output_shape[1:])

        # Training.
        train_adversarial(S, D, *data, iters_trn, iters_val, epochs,
                          batch, alpha0, alpha1, alpha_switch_epoch)
