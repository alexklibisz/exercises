import argparse
import matplotlib.pyplot as plt
import numpy as np
import pdb
import tifffile as tiff

from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from keras.losses import binary_crossentropy

import sys
sys.path.append('.')
from networks import UNet, ConvNetClassifier, set_trainable, get_trainable_count

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


def train_standard(net_seg, imgs_trn, msks_trn, imgs_val, msks_val, steps_trn, steps_val, input_shape, epochs, batch):
    """Builds and trains the segmentation network by itself."""

    # Define sample generators. Generate a single validation set.
    gen_trn = sampler(imgs_trn, msks_trn, input_shape, batch)
    gen_val = sampler(imgs_val, msks_val, input_shape, steps_val)
    x_val, y_val = next(gen_val)

    cb = [
        ModelCheckpoint('checkpoints/std_net_seg_{val_loss:.2f}.hdf5',
                        monitor='val_loss', save_best_only=1, verbose=1, mode='min'),
        CSVLogger('checkpoints/std_history.csv')
    ]

    net_seg.compile(optimizer=RMSprop(0.001, decay=1e-3), loss=binary_crossentropy)
    net_seg.fit_generator(gen_trn, epochs=epochs, steps_per_epoch=steps_trn,
                          validation_data=(x_val, y_val), callbacks=cb)


def train_dscersarial(net_seg, net_dsc, imgs_trn, msks_trn, imgs_val, msks_val, steps_trn, steps_val, input_shape, epochs, batch, alpha):
    """Builds and trains the segmentation network and adversarial classifier."""

    from keras.models import Model, model_from_json
    from keras import backend as K

    # Make the "generator", which combines the segmentation network and the discriminator.
    net_dsc_copy = model_from_json(net_dsc.to_json())
    set_trainable(net_dsc_copy, False)
    net_cmb = Model(net_seg.input, outputs=[net_seg.output, net_dsc_copy(net_seg.output)])
    ll, ww = [binary_crossentropy, binary_crossentropy], [1, alpha]
    net_cmb.compile(optimizer=RMSprop(0.001, decay=1e-3), loss=ll, loss_weights=ww)
    assert get_trainable_count(net_cmb) == get_trainable_count(net_seg)
    assert id(net_cmb.layers[-1]) == id(net_dsc_copy)
    assert id(net_cmb.layers[-1]) != id(net_dsc)

    # Compile the standalone adversarial network.
    def yp_mean(yt, yp):
        return K.mean(yp)

    net_dsc.compile(optimizer=Adam(0.001), loss=binary_crossentropy, metrics=[yp_mean])

    # Define sample generators to yield an epoch of data at once.
    gen_trn = sampler(imgs_trn, msks_trn, input_shape, batch * steps_trn)
    gen_val = sampler(imgs_val, msks_val, input_shape, batch)

    # Callbacks.
    cb_dsc = [
        TensorBoard(log_dir='checkpoints/tblogs', histogram_freq=1, batch_size=batch, write_graph=False, write_grads=True,
                    write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    ]

    # Training loop. Alternating one epoch training the combined model followed
    # by an epoch of training the adversarial model.
    for epoch in range(epochs):

        imgs_epoch, msks_epoch = next(gen_trn)

        # Train the combined model for one epoch. Freeze the adversarial classifier
        # so that gradient updates are only made to the segmentation network.
        net_cmb.layers[-1].set_weights(net_dsc.get_weights())
        w0 = np.concatenate([w.flatten() for w in net_cmb.layers[-1].get_weights()])
        x, y = imgs_epoch, [msks_epoch, np.ones((len(msks_epoch), 1))]
        net_cmb.fit(x, y, epochs=epoch + 1, batch_size=batch, initial_epoch=epoch)
        w1 = np.concatenate([w.flatten() for w in net_cmb.layers[-1].get_weights()])
        assert(np.all(w0 == w1))

        # Generate fake and real data for the discriminator.
        x_fake = net_seg.predict(imgs_epoch, batch_size=batch)
        x_real = msks_epoch
        y_fake = np.zeros((len(x_fake), 1))
        y_real = np.ones((len(x_real), 1))

        plt.imshow(np.hstack([x_fake[0, :, :, 0], x_real[0, :, :, 0]]), cmap='gray')
        plt.title('Epoch %d' % epoch)
        plt.show()

        # Train discriminator one epoch on combined fake and real data.
        x, y = np.concatenate([x_fake, x_real], axis=0), np.concatenate([y_fake, y_real], axis=0)
        net_dsc.fit(x, y, epochs=epoch + 1, batch_size=batch,
                    initial_epoch=epoch, validation_split=0.2, callbacks=cb_dsc)


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

        # Load data and split for training and validation.
        # Important to remember that the tiffs have range [0, 255].
        imgs = tiff.imread(trn_imgs_path)[:, :, :, np.newaxis] / 255.
        msks = tiff.imread(trn_msks_path)[:, :, :, np.newaxis] / 255.

        # Normalize images.
        imgs -= np.mean(imgs)
        imgs /= np.std(imgs)

        imgs_trn, msks_trn = imgs[:20, ...], msks[:20, ...]
        imgs_val, msks_val = imgs[20:, ...], msks[20:, ...]
        data = (imgs_trn, msks_trn, imgs_val, msks_val)

        # Network and training parameters.
        input_shape = (256, 256, 1)
        epochs = 40
        steps = imgs_trn.shape[0]
        batch = 4
        steps_trn = int(np.prod(imgs_trn.shape[:-1]) / np.prod((batch, *input_shape))) * 2
        steps_val = steps_trn
        alpha = 0.5

        # Define networks, sample generators, callbacks.
        net_seg = UNet(input_shape)

        if args['adversarial']:
            net_dsc = ConvNetClassifier(net_seg.output_shape[1:])
            train_dscersarial(net_seg, net_dsc, *data, steps_trn, steps_val, input_shape, epochs, batch, alpha)
        else:
            train_standard(net_seg, *data, steps_trn, steps_val, input_shape, epochs, batch)

    if args['submit']:

        pass
