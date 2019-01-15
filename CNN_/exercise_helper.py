#This module contains functions used in Exercise 3 and must be in
# the same directory as Exercise3.ipynb. These functions are identical
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

    rval = [train_set_x, valid_set_x, test_set_x]
    return rval

def conv_layer(rng, input, image_shape, filter_shape, border_mode, activation, bias):
    # Inputs:
    # rng - Random number generator
    # input - Symbolic image tensor
    # image_shape - dimensions of input image 
    #              tuple or list of length 4
    #              (batch size, depth, image height, image width)
    # filter_shape - dimensions of filter in convolution layer
    #               tuple or list of length 4:
    #               (number of channels, depth, height, width)
    # border_mode - border mode for convolution function 
    # activation - activation function to be used
    # bias - value for the initial bias 
    # No pooling layer 
    
    # Outputs:
    # output - tensor containing activations fed into next layer
    # params - list containing layer parameters

    assert image_shape[1] == filter_shape[1]

    # initialize weights with random weights 
    # and assign a shared variable to weights with samples
    # drawn from a gaussian distribution with 0 mean and standard deviation of 
    # 0.1. This is just a random initialization.
    W = theano.shared(
        np.asarray(rng.normal(loc=0, scale=0.1, size=filter_shape)),
        borrow=True)

    # convolve input with the specified type of kernel
    # Theano function takes as input the data tensor, filter weights, filter 
    # specifications, image specifications and the convolution mode. 
    # In our example, the dimensions of the output of this operation would be:
    # mini_batch_size x 9 x 24 x 24
    conv_out = conv2d(input=input, 
                      filters=W, 
                      filter_shape=filter_shape, 
                      input_shape=image_shape, 
                      border_mode=border_mode)

    # no pooling

    if bias is None:
        # if no specs for bias is given
        # bias is a 1D tensor, one bias per output feature map. 
        # initialize to zero and assign a shared variable to bias
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, borrow=True)
    else:
        # otherwise assign 1D or 2D tensor based on the initializations 
        b = theano.shared(value=bias, borrow=True)

    # before adding the bias term to the specified activation function,
    # check the dimensions of bias shared variable
    # b.dimshuffle is a broadcasting operation and
    # returns a view of the bias tensor with permuted dimensions
    # In this case, our bias tensor is originally of the dimension 9 x 1.
    if len(b.eval().shape) < 2: 
        # add the bias term. Since the bias is a 1D array tensor, we first
        # broadcast it to a tensor of shape (1, # of channels, 1, 1).
        # (1, 9, 1, 1)
        output = activation(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
    else:
        # add the bias term 
        # since the bias is an 2D tensor, we 
        # broadcast it to a tensor of shape 
        # (1, # of channels, image_shape[0], image_shape[1]); i.e,
        # (1,9,28,28)
        output = activation(conv_out + b.dimshuffle('x', 'x', 0, 1))
        
    # return output and parameters as a single list
    params = [W,b]
    return output, params
        

