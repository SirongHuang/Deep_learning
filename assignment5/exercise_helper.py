# This module contains functions used in Exercise 2 and must be in
# the same directory as Exercise2.ipynb. These functions are identical
# to the functions used in the corresponding demonstration.

import os
import six.moves.cPickle as pickle
import gzip
import numpy as np
import theano
import theano.tensor as T
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mt
import random as rd
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
from matplotlib.patches import Ellipse


def load_data(dataset, train_size=6000,binarize=None):
    # Function to load dataset. If dataset is not present in current directory
    # download from specified link

    # Inputs:
    # dataset - name of dataset
    # train_size - Number of training samples

    # Outputs:
    # rval - array containing training, validation and test set

    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(os.path.split(__file__)[0], dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('***** Loading data *****')

    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    def shared_dataset(data_xy, sample_size, borrow=True,binarize=None):
        # Seeding the random number generator
        rng = np.random.RandomState(23455)
        rd.seed(23455)

        # Function to create theano shared variables for the data

        # Inputs:
        # data_xy - dataset
        # borrow - boolean to set borrow parameter
        # sample_size - number of dataset samples

        # Outputs:
        # shared_x - The x values (image data)
        # shared_y - The class labels

        data_x, data_y = data_xy

        if binarize == "stochastic":
            data_x = data_x
            data_x = (data_x + np.random.rand(data_x.shape[0],data_x.shape[1]) > 1.0).astype(np.float32)
            # data_x = np.random.binomial(1,data_x).astype(np.float32)
            # data_x = ((data_x + np.random.rand(D)) > 1.0)
        elif binarize == "threshold":
            data_x = data_x
            data_x = (data_x > 0.5).astype(np.float32)

        indices = 0
        if (sample_size < 0):
            print('Sample size too small!')
            return
        try:
            indices = rd.sample(range(0, data_y.shape[0]), sample_size)
        except ValueError:
            print('Sample size exceeds data size.')
        data_x = data_x[indices, :]
        data_y = data_y[indices]

        shared_x = theano.shared(np.asarray(data_x,dtype=theano.config.floatX),borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,dtype=theano.config.floatX),borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set,sample_size=min(train_size//3,test_set[1].shape[0]),binarize=binarize)
    valid_set_x, valid_set_y = shared_dataset(valid_set,sample_size=min(train_size//3,valid_set[1].shape[0]),binarize=binarize)
    train_set_x, train_set_y = shared_dataset(train_set,sample_size=train_size,binarize=binarize)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),(test_set_x, test_set_y)]
    return rval


def plot_2d_proj(proj, y_vals,name=None):
    matplotlib.rcParams['figure.figsize'] = (6.0,4.0)
    plt.figure()
    colors = ["#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c","#fdbf6f","#ff7f00","#cab2d6","#6a3d9a"]
    for i,color in enumerate(colors):
        digit_class = y_vals == i
        plt.scatter(proj[digit_class,0], proj[digit_class,1], c=color, alpha=0.5, label=str(i))
    leg = plt.legend(); [l.set_alpha(1) for l in leg.legendHandles]
    if name is not None:
        plt.title(name)
    plt.show()



def plot_act_rec(actual,recon,name=None):
    matplotlib.rcParams['figure.figsize'] = (8.0,4.0)

    fig = plt.figure()
    gs = matplotlib.gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0],aspect='equal')
    im1 = ax1.imshow(actual.reshape((5,4,28,28)).swapaxes(1,2).reshape(28*5,28*4),cmap='gray')
    fig.colorbar(im1, ax=ax1,extend='max')
    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    ax1.set_title("actual")


    ax2 = fig.add_subplot(gs[0, 1],aspect='equal')
    im2 = ax2.imshow(recon.reshape((5,4,28,28)).swapaxes(1,2).reshape(28*5,28*4),cmap='gray')
    fig.colorbar(im2, ax=ax2,extend='max')
    ax2.xaxis.set_ticklabels([])
    ax2.yaxis.set_ticklabels([])
    ax2.set_title("reconstruction")

    plt.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off

    if name is not None:
        fig.suptitle(name)
    plt.show()

def gradient_updates_Adam(cost, params, learning_rate,eps = 1e-8,beta1 = 0.9,beta2 = 0.999):
    # Function to return an update list for the parameters to be updated

    # cost: MSE cost Theano variable
    # params :  parameters coming from hidden and output layers
    # learning rate: learning rate defined as hyperparameter

    # Outputs:
    # updates : updates to be made and to be defined in the train_model function.
    updates = []
    for param in params:

            t = theano.shared(1)
            s = theano.shared(param.get_value(borrow=True)*0.)
            r = theano.shared(param.get_value(borrow=True)*0.)
            updates.append((s, beta1*s + (1.0-beta1)*T.grad(cost, param)))
            updates.append((r, beta2*r + (1.0-beta2)*(T.grad(cost, param)**2)))
            s_hat =  s/(1-beta1**t)
            r_hat =  r/(1-beta2**t)
            updates.append((param, param - learning_rate*s_hat/(np.sqrt(r_hat)+eps) ))
            updates.append((t, t+1))

    return updates


def plot_latent_space2D(latent_mu,latent_logvar,y_labels,name=None):
    matplotlib.rcParams['figure.figsize'] = (8.0, 8.0)

    NUM = latent_mu.shape[0]
    latent_sd = np.exp(0.5*latent_logvar)

    colors = ["#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c","#fdbf6f","#ff7f00","#cab2d6","#6a3d9a"]
    ells = [Ellipse(xy=latent_mu[i], width=latent_sd[i,0], height=latent_sd[i,1],
                    color=colors[y_labels[i]],label=str(y_labels[i]))
            for i in range(NUM)]

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.3)
        e.set(label = e.get_label())

    _,label_idx = np.unique(y_labels, return_index=True)
    ax.legend([ells[e] for e in label_idx], ['{}'.format(y_labels[i]) for i in label_idx])
    ax.set_xlim(latent_mu.min(axis=0)[0]-1,latent_mu.max(axis=0)[1]+1)
    ax.set_ylim(latent_mu.min(axis=0)[0]-1,latent_mu.max(axis=0)[1]+1)
    if name is not None:
        ax.set_title(name)
    plt.show()
    
    
def plot_activations(activations,y_sorted_lables,name=None):
    matplotlib.rcParams['figure.figsize'] = (8.0,5.0)
    fig,ax = plt.subplots()
    im = ax.imshow(activations.T,cmap="RdBu")
    ax.set_xticks(np.linspace(0,y_sorted_lables.shape[0],20))
    ax.set_xticklabels(y_sorted_lables[np.linspace(0,y_sorted_lables.shape[0]-1, 20).astype(int)])
    ax.set_yticklabels([])
    ax.set_ylabel("hidden units")
    ax.set_xlabel("labels")
    fig.colorbar(im, ax=ax,extend='max',fraction=0.046, pad=0.04)
    if name is not None:
        ax.set_title(name)
    plt.show()
    
    
def plot_img(weights,name=None):
    matplotlib.rcParams['figure.figsize'] = (6.,6.0)
    fig = plt.figure()
    gs = matplotlib.gridspec.GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0],aspect='equal')
    im1 = ax1.imshow(weights.reshape((5,4,28,28)).swapaxes(1,2).reshape(28*5,28*4),cmap='gray')
    fig.colorbar(im1, ax=ax1,extend='max')
    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    if name is not None:
        ax1.set_title(name)
    plt.show()
