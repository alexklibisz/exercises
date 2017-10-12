import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pdb
import tifffile as tiff

from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping
from keras.losses import binary_crossentropy as bce
from keras.models import Model, model_from_json
from keras import backend as K

import sys
sys.path.append('.')
from networks import UNet, ConvNetClassifier, set_trainable, get_trainable_count, get_flat_weights

np.random.seed(865)


def sampler(imgs, msks, input_shape, batch):
    """Generator that yields training samples randomly sampled from given data
    with simple augmentations."""
    N, H, W = imgs.shape[:-1]
    h, w = input_shape[:-1]
    ii = np.arange(N)
    transforms = [
        lambda x: x,
        lambda x: np.rot90(x, 1, (0, 1)),
        lambda x: np.rot90(x, 2, (0, 1)),
        lambda x: np.rot90(x, 3, (0, 1)),
    ]
    while True:
        ii_ = np.random.choice(ii, batch, replace=(N <= batch))
        y0 = np.random.randint(0, H - h)
        x0 = np.random.randint(0, W - w)
        imgs_batch = imgs[ii_, y0:y0 + h, x0:x0 + h, ...]
        msks_batch = msks[ii_, y0:y0 + h, x0:x0 + h, ...]
        tt = np.random.choice(transforms, batch)
        for i, t in enumerate(tt):
            imgs_batch[i] = t(imgs_batch[i])
            msks_batch[i] = t(msks_batch[i])
        yield imgs_batch, msks_batch


def train_standard(S, imgs_trn, msks_trn, imgs_val, msks_val, steps_trn, steps_val, input_shape, epochs, batch):
    """Builds and trains the segmentation network by itself."""

    # Define sample generators. Generate a single validation set.
    gen_trn = sampler(imgs_trn, msks_trn, input_shape, batch)
    gen_val = sampler(imgs_val, msks_val, input_shape, steps_val)
    x_val, y_val = next(gen_val)

    cb = [
        ModelCheckpoint('checkpoints/std_S_{val_loss:.2f}.hdf5',
                        monitor='val_loss', save_best_only=1, verbose=1, mode='min'),
        CSVLogger('checkpoints/std_history.csv')
    ]

    S.compile(optimizer=RMSprop(0.001, decay=1e-3), loss=bce)
    S.fit_generator(gen_trn, epochs=epochs, steps_per_epoch=steps_trn,
                    validation_data=(x_val, y_val), callbacks=cb)


def train_adversarial(S, D, imgs_trn, msks_trn, imgs_val, msks_val, steps_trn, steps_val, input_shape, epochs, batch, alpha):
    """Builds and trains the segmentation network and adversarial classifier."""

    S.summary()
    D.summary()

    # Combine segmentation with frozen discriminator to make generator.
    set_trainable(D, False)
    G = Model(S.input, outputs=[S.output, D(S.output)])
    G.compile(optimizer=RMSprop(0.001, decay=1e-3),
              loss=bce, loss_weights=[1, alpha])
    assert get_trainable_count(G) == get_trainable_count(S)
    assert id(G.layers[-1]) == id(D)

    # G.load_weights('checkpoints/std_net_seg_0.20.hdf5', by_name=True)

    # Compile the standalone adversarial network.
    def yp_mean(yt, yp):
        return K.mean(yp)

    # It seems to be important that the learning rate is not too high. lr 0.001
    # makes the discriminator immediately predict all positive or all negative.
    set_trainable(D, True)
    D.compile(optimizer=Adam(0.0001), loss=bce, metrics=['accuracy', yp_mean])

    # Define sample generators to yield an epoch of data at once.
    gen_trn = sampler(imgs_trn, msks_trn, input_shape, batch * steps_trn)
    gen_val = sampler(imgs_val, msks_val, input_shape, batch * steps_val)

    # Single validation set.
    imgs_epoch_val, msks_epoch_val = next(gen_val)

    # Callbacks.
    cb_dsc = [
        TensorBoard(log_dir='checkpoints/tblogs', histogram_freq=1,
                    batch_size=batch, write_graph=False, write_grads=True),
        EarlyStopping(monitor='val_loss', min_delta=0, patience=1, mode='min')
    ]

    # Pre-computed labels for generator and discriminator.
    gyt = np.ones((batch * steps_trn)) * 1.0    # Generator y-train.
    gyv = np.ones((batch * steps_val)) * 1.0    # Generator y-val.
    dytf = np.ones((batch * steps_trn)) * 0.0   # Discriminator y-train-fake.
    dytr = np.ones((batch * steps_trn)) * 1.0   # Discriminator y-train-real.
    dyvf = np.ones((batch * steps_val)) * 0.0   # Discriminator y-val-fake.
    dyvr = np.ones((batch * steps_val)) * 1.0   # Discriminator y-val-real.

    # Training loop.
    for ep in range(epochs):

        # New training set at each epoch.
        imgs_epoch_trn, msks_epoch_trn = next(gen_trn)

        # Train the combined model for one epoch. Freeze the adversarial classifier
        # so that gradient updates are only made to the segmentation network.
        G.layers[-1].set_weights(D.get_weights())
        xt, yt = imgs_epoch_trn, [msks_epoch_trn, gyt]
        xv, yv = imgs_epoch_val, [msks_epoch_val, gyv]
        G.fit(xt, yt, epochs=ep + 1, batch_size=batch,
              initial_epoch=ep, validation_data=(xv, yv))

        # Generate fake and real data for the discriminator.
        xtf, xtr = S.predict(imgs_epoch_trn, batch_size=batch), msks_epoch_trn
        xvf, xvr = S.predict(imgs_epoch_val, batch_size=batch), msks_epoch_val

        plt.hist(xtf.flatten(), color='red', alpha=0.3)
        plt.hist(xtr.flatten(), color='blue', alpha=0.3)
        plt.savefig('out.png')

        s = [np.hstack([xvf[i, :, :, 0], xvr[i, :, :, 0]]) for i in range(3)]
        plt.imshow(np.vstack(s), cmap='gray')
        plt.title('Epoch %d' % ep)
        plt.savefig('checkpoints/adv_sample_%02d.png' % (ep))

        # Combine real and fake data to train discriminator.
        xt, yt = np.concatenate([xtf, xtr]), np.concatenate([dytf, dytr])
        xv, yv = np.concatenate([xvf, xvr]), np.concatenate([dyvf, dyvr])

        # Train discriminator.
        # ww1 = get_flat_weights(D)
        D.fit(xt, yt, epochs=ep + 1, batch_size=batch, callbacks=cb_dsc,
              initial_epoch=ep, validation_data=(xv, yv))
        # ww2 = get_flat_weights(D)
        # print(np.allclose(ww1, ww2))

        # pdb.set_trace()


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
        imgs_trn, msks_trn = imgs[:24, ...], msks[:24, ...]
        imgs_val, msks_val = imgs[24:, ...], msks[24:, ...]
        data = (imgs_trn, msks_trn, imgs_val, msks_val)

        # Network and training parameters.
        input_shape = (96, 96, 1)
        epochs = 40
        steps = imgs_trn.shape[0]
        batch = 4
        steps_trn = int(np.prod(imgs_trn.shape[:-1]) / np.prod((batch, *input_shape)))
        steps_val = int(np.prod(imgs_val.shape[:-1]) / np.prod((batch, *input_shape)))
        alpha = 1.0

        # Define networks, sample generators, callbacks.
        S = UNet(input_shape)

        if args['adversarial']:
            D = ConvNetClassifier(S.output_shape[1:])
            train_adversarial(S, D, *data, steps_trn, steps_val, input_shape, epochs, batch, alpha)
        else:
            train_standard(S, *data, steps_trn, steps_val, input_shape, epochs, batch)

    if args['submit']:

        pass
