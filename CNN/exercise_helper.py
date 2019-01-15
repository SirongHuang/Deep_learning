# This module contains functions used in Exercise 2 and must be in
# the same directory as Exercise2.ipynb. These functions are identical
# to the functions used in the corresponding demonstration.

import os
import six.moves.cPickle as pickle
import gzip
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mt
import random as rd
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

# Seeding the random number generator
rng = np.random.RandomState(23455)
rd.seed(23455)

def load_data(dataset, train_size=6000):
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

    def shared_dataset(data_xy, sample_size, borrow=True):
        # Function to create theano shared variables for the data

        # Inputs:
        # data_xy - dataset
        # borrow - boolean to set borrow parameter
        # sample_size - number of dataset samples

        # Outputs:
        # shared_x - The x values (image data)
        # shared_y - The class labels

        data_x, data_y = data_xy
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

        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(
                                            test_set,
                                            sample_size=train_size//3)
    valid_set_x, valid_set_y = shared_dataset(
                                            valid_set,
                                            sample_size=train_size//3)
    train_set_x, train_set_y = shared_dataset(
                                            train_set,
                                            sample_size=train_size)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def pooling(input_pool, size):
    # Function to perform Max-pooling on each feature map

    # Inputs:
    # input_pool - feature maps obtained as output from convolution layer.
    # size - specification of downsampling (pooling) factor.
    #        tuple format: (# of rows, # of columns)

    # Outputs:
    # pool_out - pooled output.
    #            dimensions: (# of channels, conv_output_height/#rows,
    #                         conv_output_width/#rows)

    pool_out = pool.pool_2d(input=input_pool, ws=size, ignore_border=True)
    return pool_out


def convLayer(rng, data_input, filter_spec, image_spec, pool_size, activation):
    # Function that defines the convolution layer. Calls the
    # pooling function and then activation function.

    # Inputs:
    # rng - random number generator used to initialize weights.
    # data_input - symbolic input image tensor.
    # filter_spec - dimensions of filter in convolution layer.
    #               tuple format:(# of channels, depth, width, height)
    # image_spec - specifications of input images.
    #              tuple format:(batch size, color channels, height, width)
    # pool_size - specification of downsampling (pooling) factor.
    #             tuple format: (# of rows, # of columns)
    # activation - activation function to be used.

    # Outputs:
    # output - tensor containing activations fed into next layer.
    # params - list containing layer parameters

    # Creating a shared variable for weights that are initialised with samples
    # drawn from a gaussian distribution with 0 mean and standard deviation of
    # 0.1. This is just a random initialisation.
    W = theano.shared(
        np.asarray(rng.normal(loc=0, scale=0.1, size=filter_spec)),
        borrow=True)

    # Bias is a 1 D tensor -- one bias per output feature map
    b = theano.shared(np.zeros((filter_spec[0],)), borrow=True)

    # Convolve input with specifications
    conv_op_out = conv2d(
        input=data_input,
        filters=W,
        filter_shape=filter_spec,
        input_shape=image_spec)

    # Add the bias term and use the specified activation function/
    # non-linearity.
    # b.dimshuffle returns a view of the bias tensor with permuted dimensions.
    # In this case our bias tensor is originally of the dimension 9 x 1. The
    # dimshuffle operation used below, broadcasts this into a tensor of
    # 1 x 9 x 1 x 1. Note that there is one bias per output feature map.
    layer_activation = activation(conv_op_out + b.dimshuffle('x', 0, 'x', 'x'))

    # Perform pooling on the activations. It is required to reduce the spatial
    # size of the representation to reduce the number of parameters and
    # computation in the network. Hence, it helps to control overfitting
    # Output dimensions: (# channels, image height-filter height+1,
    #                     image width - filter width+1)
    # In our demo, the dimensions would be of mini_batch_size x 9 x 12 x 12
    output = pooling(input_pool=layer_activation, size=pool_size)
    params = [W, b]
    return output, params


def fullyConnectedLayer(data_input, num_in, num_out):
    # Function to perform the final logistic regression using the output
    # from the hidden layer (Softmax layer). It classifies the values of
    # the fully-connected layer.

    # Inputs:
    # data_input - input for the softmax layer.
    #              (Symbolic tensor)
    # num_in - number of input units or dimensionality  of input
    # num_out - number of output units or number of output labels.

    # Outputs:
    # p_y_given_x - class-membership probabilities.
    # y_pred - class with maximal probability
    # params - parameters of the layer

    # Creating a shared variable for weights that are initialised with samples
    # drawn from a gaussian distribution with 0 mean and standard deviation of
    # 0.1. This is just a random initialisation.
    W = theano.shared(
        value=np.asarray(
            rng.normal(loc=0, scale=0.1, size=(num_in, num_out))),
        name='W',
        borrow=True)

    # Creating a shared variable for biases that are initialised with
    # zeros.
    b = theano.shared(
        value=np.zeros((num_out,)),
        name='b',
        borrow=True)

    # compute class-membership probabilities
    p_y_given_x = T.nnet.softmax(T.dot(data_input, W) + b)

    # Class prediction. Find class whose probability is maximal
    y_pred = T.argmax(p_y_given_x, axis=1)
    params = [W, b]
    return p_y_given_x, y_pred, params


def negative_log_lik(y, p_y_given_x):
    # Function to compute the cost that is to be minimised.
    # Here, we compute the negative log-likelihood.

    # Inputs:
    # y - expected class label
    # p_y_given_x - class-membership probabilities

    # Outputs:
    # cost_log - the computed negative log-lik cost

    cost_log = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
    return cost_log


def errors(y, y_pred):
    # Function to compute to the number of wrongly classified
    # instances.

    # Inputs:
    # y - expected class label
    # y_pred - predicted class label

    # Outputs:
    # count_error - number of wrongly classified instances

    count_error = T.mean(T.neq(y_pred, y))
    return count_error


def generate_plot(cost, iter, valid_score, epoch, weights):
    # Function to generate plots for cost, prediction error and visualisation
    # of convolution layer weights.

    # Inputs:
    # cost - array containing cost per iteration
    # iter - array of iterations
    # valid_score - array containing rate of prediction errors
    # epoch - array of epochs
    # weights - weights of the first convolutional layer

    # Outputs:
    # Generates plots for the data.

    mt.rcParams['figure.figsize'] = (11, 9)
    fig = plt.figure()

    # grid spec for full plot
    gs0 = gridspec.GridSpec(1, 2)

    # grid spec for cost and accuracy
    gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[0, 0])

    # grid spec for conv layers
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0, 1])

    # sub grid specs for each visualisation
    subgs = []
    subgs.append(gridspec.GridSpecFromSubplotSpec(
                                                 weights.shape[0],
                                                 1,
                                                 subplot_spec=gs1[0, 0]))

    # accuracy plot
    ax_00 = fig.add_subplot(gs00[0, 0])
    ax_00.plot(epoch, valid_score, 'b-')
    ax_00.set_xlim(0, epoch[-1])
    ax_00.set_ylim(0, 1.1)
    ax_00.set_xlabel('Epochs')
    ax_00.set_ylabel('Prediction error')
    ax_00.set_title('Prediction Error vs. Number of Epochs')

    # cost plot
    ax_01 = fig.add_subplot(gs00[1, 0])
    ax_01.plot(iter, cost, 'r-')
    ax_01.set_xlabel('Iterations')
    ax_01.set_ylabel('Cost')
    ax_01.set_title('Cost vs. Iterations')
    ax_01.set_xlim(0, iter[-1])
    ax_01.set_ylim(0, np.max(cost))

    # normalisation
    weights = np.array(weights[:, 0])
    min_channel = np.min(weights)
    max_channel = np.max(weights)
    a = 255/(max_channel - min_channel)
    b = 255 - a * max_channel
    weights = a * weights + b

    # plot weight visualisations
    for channel_num in range(weights.shape[0]):
        ax = fig.add_subplot(subgs[0][channel_num, 0], aspect='equal')
        channel = weights[channel_num]
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(channel,
                  cmap='gray',
                  interpolation='None')
        if channel_num == 0:
            ax.set_title("Visualisation of Weights in" + "\n" +
                         "First Convolutional Layer")

    fig.tight_layout()
