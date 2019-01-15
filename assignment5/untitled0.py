
# theano imports
import theano
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# plotting functionalities
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# load pretained objects
import six.moves.cPickle as pickle

# helper functions
from exercise_helper import load_data,gradient_updates_Adam
from exercise_helper import plot_2d_proj,plot_act_rec,plot_latent_space2D
from exercise_helper import plot_img, plot_activations

theano.config.floatX = "float32"

# random number generators
rng_np = np.random.RandomState(23455)
srng = RandomStreams(seed=1234)

# change the train_size to take subsets of the data
datasets = load_data('mnist.pkl.gz', train_size = 5000,binarize="stochastic")
train_x, train_y = datasets[0]
valid_x, valid_y = datasets[1]
test_x, test_y   = datasets[2]

# implementing the VAE in theano
# similar network architecture to (Kingma and Welling 2013)

# symbolic variable declaration
input = T.matrix("input") # observation matrix as input 
learning_rate = T.scalar("learning_rate") # learning rate for the optimizer
mbidxs = T.lvector("mbidxs") # mini-batch index



###################################################################
###################### INSERT YOUR CODE HERE ######################
###################################################################

# define number of latent dimensions 
# use 2 latent dimensions for the implementation
latent_dim = 2
# a list of layer dimensions
dims = [784, 500, latent_dim, 500, 784]



# ------------------------------------------------------
# parameter initializations 
# ------------------------------------------------------

# Use Glorot and Bengio (2010) initialization for weights
# initialize weights for input to hidden layer
# size : (dims[0],dims[1])
W1 = theano.shared(rng_np.uniform(-np.sqrt(6.0/(dims[0]+dims[1])), np.sqrt(6.0/(dims[0]+dims[1])), 
                                  size=dims[0:2]), borrow=True, name="W1")                   

# initialize weights for hidden to latent layer (mean)
# size : (dims[1],dims[2])
W2_mu = theano.shared(rng_np.uniform(-np.sqrt(6.0/(dims[1]+dims[2])), np.sqrt(6.0/(dims[1]+dims[2])), 
                                  size=dims[1:3]), borrow=True, name="W2_mu")                   

# initialize weights for hidden to latent layer (log variance)
# size : (dims[1],dims[2])
# note that in order to make variance parameter unconstrained
# we choose to model log(variacne) instead of variance
W2_logvar = theano.shared(rng_np.uniform(-np.sqrt(6.0/(dims[1]+dims[2])), np.sqrt(6.0/(dims[1]+dims[2])), 
                                  size=dims[1:3]), borrow=True, name="W2_logvar") 

# initialize weights for latent layer (z) to second hidden layer 
# size : (dims[2],dims[3])
W3 = theano.shared(rng_np.uniform(-np.sqrt(6.0/(dims[2]+dims[3])), np.sqrt(6.0/(dims[2]+dims[3])), 
                                  size=dims[2:4]), borrow=True, name="W3") 

# initilize weights from second hidden layer to output layer f(z)
# size : (dims[3],dims[4])
W4 = theano.shared(rng_np.uniform(-np.sqrt(6.0/(dims[3]+dims[4])), np.sqrt(6.0/(dims[3]+dims[4])), 
                                  size=dims[3:5]), borrow=True, name="W4") 

# initialize bias for first hidden layer
# size : (dims[1],)
b1 = theano.shared(np.zeros((dims[1],)), borrow=True, name="b1")

# initialize bias for latent layer (mean)
# size : (dims[2],)
b2_mu = theano.shared(np.zeros((dims[2],)), borrow=True, name="b2_mu")

# initialize bias for latent layer (log variance)
# size : (dims[2],)
b2_logvar = theano.shared(np.zeros((dims[2],)), borrow=True, name="b2_logvar")

# initialize bias for the second hidden layer
# size : (dims[3],)
b3 = theano.shared(np.zeros((dims[3],)), borrow=True, name="b3")

# initialize bias for the output layer 
# size : (dims[4],) 
b4 = theano.shared(np.zeros((dims[4],)), borrow=True, name="b4")


# ------------------------------------------------------
# define feed-forward computations
# ------------------------------------------------------

# compute non-linear activations in the first hidden layer
# hint : use input, W1, and b1
x1     = T.tanh(T.dot(input, W1) + b1)

# compute linear activation for latent mean
# hint : use x1, W2_mu, b2_mu
mu     = T.dot(x1, W2_mu) + b2_mu 

# compute linear activation for latent logvar
# hint : use x1, W2_logvar, b2_logvar
logvar = T.dot(x1, W2_logvar) + b2_logvar

# compute z
# z = mu + {exp(0.5*logvar) * rnd_sample}
# where rnd_sample is from standard normal
# hint : pick samples of size (num_observation,dims[2])
# hint : use srng for reproducibility 
z      = mu + (T.exp(0.5*logvar) * srng.normal((mb_size,dims[2])))

# compute non-linear activation at second hidden layer
# hint: use z, W3, b3
x1_rec = T.tanh(T.dot(z, W3) + b3)

# compute networks ouputs f(z)
# hint : add non-linearity to scale output btw 0-1
# hint : use x_rec, W4, b4
x_rec  = T.nnet.sigmoid(T.dot(x1_rec, W4) + b4)


# ------------------------------------------------------
# back propagation computations
# ------------------------------------------------------

# compute lower_bound = reconstruction cost - KL cost
# minimie negative of lower bound i.e. cost = -lower_bound

# hint reconstruction cost for each observation 
# can be computed as sum of binary cross entropy 
# across all dimensions 
cost_rec = T.sum(-T.nnet.binary_crossentropy(x_rec, input)) / mb_size

# hint KL cost for each observation 
# can be computed using the formula given above
cost_KL  = (-0.5 / mb_size) * (mu.shape[0]*mu.shape[1] + T.sum(2.* logvar) -
                              T.sum(T.square(mu)) - T.sum(T.exp(2. * logvar))) / mb_size
# compute lower bound for each observation
# as reconstruction - kl cost
lower_bound = cost_rec - cost_KL

# final cost function scalar to minimize
# can be computed as negative mean of lower bound
cost = -T.mean(lower_bound)



# create a list of all the parameters to be optimized
params = [W1,W2_mu,W2_logvar,W3,W4,b1,b2_mu,b2_logvar,b3,b4]


# pass cost and params list to 
# gradient_updates_Adam for optimization
# above function returns updates list for training
# refer to assignment3 for more details
updates = gradient_updates_Adam(cost,params,learning_rate,eps = 1e-8,beta1 = 0.9,beta2 = 0.99)


# ------------------------------------------------------
# define theano functions
# ------------------------------------------------------


# define vae_train function 
# inputs : minibatch-index and learning rate
# ouputs : cost, cost_rec and cost_KL
# updates: updates from Adam
# givens : symobolic input as subset of 
#        : train_x over minibatch index
vae_train = theano.function(
        inputs = [mbidxs,learning_rate],
        outputs = [cost, cost_rec, cost_KL], 
        updates = updates,
        givens = {input : train_x[mbidxs,:],}
    )

# define vae_cost function
# inputs : input observation matrix
# outputs: cost, cost_rec and cost_KL
vae_cost = theano.function(
        inputs = [input],
        outputs = [cost, cost_rec, cost_KL]
    )

# define vae_predict function
# inputs : input observation matrix
# outputs: f(z), latent mu and logvar
vae_predict = theano.function(
        inputs = [input],
        outputs = [x_rec, mu, logvar]
    )


# define vae_generate function
# inputs : input z
# outputs: f(z) or output of the final layer
vae_generate = theano.function(
        inputs = [z],
        outputs = [x_rec]
    )

# define vae_z function
# inputs : input observations
# outputs: z or latent space for the input
vae_z = theano.function(
        inputs = [input],
        outputs = [z]
    )



###################################################################
############################ END ##################################
###################################################################


# define training parameters
mb_size = 50
epochs = 20
lrate = 0.001
train_costs = []
valid_costs = []
train_start_time = time.time()

train_size    = len(train_y.eval()) 
training_idxs = np.arange(train_size)

print("***** training started  *****")
for epoch in range(epochs):
    
    rng_np.shuffle(training_idxs)
                
    try:    
        # initialize empty lists to collect costs
        t_minib_costs = []
        t_minib_cost_kl = []
        t_minib_cost_rec = []
        start_time = time.time()
        
        for batch_startidx in range(0, train_size, mb_size):

            mini_batch = training_idxs[batch_startidx:batch_startidx+mb_size]
            
            # MODEL TRAINING CALL
            mb_train_cost, mb_cost_rec,mb_cost_kl  = vae_train(
                    mbidxs = np.asarray(mini_batch).astype(np.int32),
                    learning_rate=lrate)
            
            # collect costs
            t_minib_costs.append(np.mean(mb_train_cost))
            t_minib_cost_kl.append(np.mean(mb_cost_kl))
            t_minib_cost_rec.append(np.mean(mb_cost_rec))
            
            
        valid_cost, _, _ = vae_cost(input = valid_x.get_value())
        
        train_costs.append(np.mean(t_minib_costs))
        valid_costs.append(np.mean(valid_cost))
        
        print("epoch={:<3d} -- train_cost={:>6.4f} -- valid_cost={:>6.4f} -- cost(Rec,KL)=({:>6.4f},{:>6.4f}) -- time={:>5.3f}".format(epoch+1, train_costs[-1], valid_costs[-1], np.mean(t_minib_cost_rec),np.mean(t_minib_cost_kl),time.time()-start_time))    
    
    except KeyboardInterrupt as e:
        print("***** stopping *****")
        break
        