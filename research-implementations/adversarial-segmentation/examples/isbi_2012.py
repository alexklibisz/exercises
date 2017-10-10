import argparse
import matplotlib.pyplot as plt
import numpy as np
import pdb
import tifffile as tiff

from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.losses import binary_crossentropy

import sys
sys.path.append('.')
from networks import UNet, ConvNetClassifier, set_trainable

np.random.seed(865)


def sampler(imgs, msks, input_shape, batch):
    """Generator that yields training samples by randomly sampling windows from
    given data. Masks get one-hot encoded."""
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
        msks_batch = np.zeros((*imgs_batch.shape[:-1], 2))
        msks_batch[:, :, :, 1:] = msks[ii_, y0:y0 + h, x0:x0 + w, ...]
        msks_batch[:, :, :, :1] = 1 - msks[ii_, y0:y0 + h, x0:x0 + w, ...]
        for i in range(batch):
            t = np.random.choice(transforms)
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


def train_adversarial(imgs_trn, msks_trn, imgs_val, msks_val, net_seg, net_adv, input_shape, epochs, batch, alpha):
    """Builds and trains the segmentation network and adversarial classifier."""

    from keras.models import Model
    from keras import backend as K

    # Assemble and compile a network that combines the segmentation and adversarial
    # classification as a single trainable network.
    net_cmb = Model(net_seg.input, outputs=[net_seg.output, net_adv(net_seg.output)])
    ll, ww = [binary_crossentropy, binary_crossentropy], [1, alpha]
    net_cmb.compile(optimizer=Adam(0.0007), loss=ll, loss_weights=ww)

    # The adv net in the combined model is still the *same* instance as the standalone.
    assert id(net_cmb.layers[-1]) == id(net_adv)

    # Compile the standalone adversarial network.
    def YP0(yt, yp):
        return K.mean(yp[:, 0])

    def YP1(yt, yp):
        return K.mean(yp[:, 1])

    # Define sample generators.
    steps_trn = int(np.prod(imgs_trn.shape) / np.prod((batch, *input_shape)))
    gen_trn = sampler(imgs_trn, msks_trn, input_shape, steps_trn, batch)
    steps_val = int(np.prod(imgs_val.shape) / np.prod((batch, *input_shape)))
    gen_val = sampler(imgs_val, msks_val, input_shape, steps_val, batch)

    # Adv targets for real and fake inputs. argmax = 0 -> fake, argmax = 1 -> real.
    o, z = np.ones((batch, 1)), np.zeros((batch, 1))
    target_real = np.hstack([z, o])
    target_fake = np.hstack([o, z])

    # Pre-train segmentation model.
    net_seg.compile(optimizer=Adam(0.0005), loss=binary_crossentropy)
    net_seg.fit_generator(gen_trn, epochs=10, steps_per_epoch=steps_trn,
                          validation_data=gen_val, validation_steps=steps_val)

    N = 10000
    X = np.zeros((N, *input_shape[:-1], 2), dtype=np.float32)
    Y = np.zeros((N, 2), dtype=np.uint8)
    normal = np.random.normal
    for i in range(0, N // 2, 2 * batch):
        imgs_batch, msks_batch_real = next(gen_trn)
        msks_batch_fake = net_seg.predict(imgs_batch)
        noise = normal(0, 0.05, imgs_batch.shape)
        X[i:i + batch] = np.clip(msks_batch_real, 0.2, 0.8) + noise
        # X[i:i + batch] = msks_batch_real
        Y[i:i + batch] = target_real
        X[i + batch:i + 2 * batch] = np.clip(msks_batch_fake.round(), 0.2, 0.8) + noise
        # X[i + batch:i + 2 * batch] = msks_batch_fake
        Y[i + batch:i + 2 * batch] = target_fake

    fig, _ = plt.subplots(2, 10, figsize=(20, 5))
    for i in range(20):
        fig.axes[i].imshow(X[i, :, :, 0], cmap='gray')
        fig.axes[i].imshow(X[i, :, :, 0], cmap='gray')

    plt.show()

    net_adv.compile(optimizer=Adam(0.0005), loss=binary_crossentropy, metrics=[YP0, YP1])
    net_adv.fit(X, Y, batch_size=batch, shuffle=True, epochs=epochs, validation_split=0.2)
    pdb.set_trace()

    # # Training loop.
    # for epoch in range(epochs):
    #     print('=== Epoch %d ===' % (epoch))
    #     for step in range(steps_trn):
    #         # Next random batch.
    #         imgs_batch, msks_batch = next(gen_trn)
    #
    #         x_real = np.clip(msks_batch, 0.2, 0.8) + np.random.normal(0, 0.05, msks_batch.shape)
    #         x_fake = net_seg.predict(imgs_batch)
    #         x = np.concatenate([x_real, x_fake], axis=0)
    #         y = np.concatenate([target_real, target_fake], axis=0)
    #         loss_adv = net_adv.train_on_batch(x, y)
    #         loss_adv = net_adv.evaluate(x, y, verbose=0, batch_size=batch)
    #
    #         s = ' '.join(['%s=%.3lf' % (k, v) for k, v in zip(net_adv.metrics_names, loss_adv)])
    #         print('adv: %s' % s)
    #
    #         # if epoch % 5 == 0:
    #         #     loss_adv = net_adv.evaluate(x, y)
    #         #     s = ' '.join(['%s=%.3lf' % (k, v) for k, v in zip(net_adv.metrics_names, loss_adv)])
    #         #     print('adv: %s' % s)
    #
    #         # # Train classifier on batch of segmentation outputs, should predict "fake".
    #         # x_fake = net_seg.predict(imgs_batch)
    #         # loss_adv_fake = net_adv.train_on_batch(x_fake, target_fake)
    #
    #         # plt.hist(x_real.flatten(), color='blue', alpha=0.3)
    #         # plt.hist(x_fake.flatten(), color='red', alpha=0.3)
    #         # plt.show()
    #
    #         # Train the combined network. Adv layers are frozen. Seg output optimized to match
    #         # ground-truth masks. Adv takes seg output as input, optimized to predict "real".
    #         # set_trainable(net_adv, False)
    #         # loss_cmb = net_cmb.train_on_batch(imgs_batch, [msks_batch, target_real])
    #         # set_trainable(net_adv, True)
    #
    #         # s1 = ' '.join(['%s=%.3lf' % (k, v) for k, v in zip(net_cmb.metrics_names, loss_cmb)])
    #         # s2 = ' '.join(['%s=%.3lf' % (k, v) for k, v in zip(net_adv.metrics_names, loss_adv_fake)])
    #         # s3 = ' '.join(['%s=%.3lf' % (k, v) for k, v in zip(net_adv.metrics_names, loss_adv_real)])
    #         # print('cmb: %s | adv fake: %s | adv real: %s' % (s1, s2, s3))
    #
    #         # s2 = ' '.join(['%s=%.3lf' % (k, v) for k, v in zip(net_adv.metrics_names, loss_adv_real)])
    #         # s3 = ' '.join(['%s=%.3lf' % (k, v) for k, v in zip(net_adv.metrics_names, loss_adv_fake)])
    #         # print('adv real: %s | adv fake: %s' % (s2, s3))


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('--train', action='store_true', default=True)
    ap.add_argument('--adversarial', action='store_true', default=False)
    ap.add_argument('--submit', action='store_true', default=False)
    ap.add_argument('--model', type=str)
    ap.add_argument('--data_dir', type=str, default='data/isbi_2012')
    args = vars(ap.parse_args())

    # Paths for serializing model, reading data, etc.
    seg_model_path = 'artifacts/isbi_2012_seg.hdf5'
    adv_model_path = 'artifacts/isbi_2012_adv.hdf5'
    trn_imgs_path = 'data/isbi_2012/train-volume.tif'
    trn_msks_path = 'data/isbi_2012/train-labels.tif'
    tst_imgs_path = 'data/isbi_2012/test-volume.tif'
    tst_msks_path = 'data/isbi_2012/test-labels.tif'

    if args['train']:

        # Load data and split for training and validation.
        # Important to remember that the tiffs have range [0, 255].
        imgs = tiff.imread(trn_imgs_path)[:, :, :, np.newaxis] / 255.
        msks = tiff.imread(trn_msks_path)[:, :, :, np.newaxis] / 255.
        imgs -= np.mean(imgs)
        imgs /= np.std(imgs)

        imgs_trn, msks_trn = imgs[:20, ...], msks[:20, ...]
        imgs_val, msks_val = imgs[20:, ...], msks[20:, ...]
        data = (imgs_trn, msks_trn, imgs_val, msks_val)

        # Network and training parameters.
        nb_classes_seg = len(np.unique(msks_trn))   # Number of segmentation labels.
        nb_classes_adv = 2                          # Real or Fake segmentation mask.
        input_shape = (256, 256, 1)                 # Sample shape.
        epochs = 20
        steps = imgs_trn.shape[0]
        batch = 4
        steps_trn = int(np.prod(imgs_trn.shape[:-1]) / np.prod((batch, *input_shape))) * 2
        steps_val = steps_trn
        alpha = 1.

        # Define networks, sample generators, callbacks.
        net_seg = UNet(input_shape, nb_classes_seg)

        if args['adversarial']:
            net_adv = ConvNetClassifier(net_seg.output_shape[1:])
            train_adversarial(net_seg, net_adv, *data, steps_trn, steps_val, input_shape, epochs, batch, alpha)
        else:
            train_standard(net_seg, *data, steps_trn, steps_val, input_shape, epochs, batch)

    if args['submit']:

        pass

    import pdb
    pdb.set_trace()

    # Load data.
