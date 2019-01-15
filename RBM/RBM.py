import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class RBM(object):
    """Restricted Boltzmann Machine Class (RBM)  """
    def __init__(
        self,
        input=None,
        n_visible=None,
        n_hidden=None,
        W=None,
        hbias=None,
        vbias=None,
        numpy_rng=None,
        theano_rng=None
    ):
        
        # RBM constructor
        # Defines the parameters of the model along with
        # basic operations for inferring hidden from visible (and vice-versa),
        # as well as for performing CD updates.

        # Inputs:
        # input: None for standalone RBMs or symbolic variable if RBM is
        # part of a larger graph

        # n_visible: number of visible units

        # n_hidden: number of hidden units

        # W: None for standalone RBMs or symbolic variable pointing to a
        # shared weight matrix in case RBM is part of a DBN network; in a DBN,
        # the weights are shared between RBMs and layers of a MLP

        # hbias: None for standalone RBMs or symbolic variable pointing
        # to a shared hidden units bias vector in case RBM is part of a
        # different network

        # vbias: None for standalone RBMs or a symbolic variable
        # pointing to a shared visible units bias

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            # theano shared variables for weights and biases
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='hbias',
                borrow=True
            )

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='vbias',
                borrow=True
            )

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        # put weights and biases to a self.params list
        self.params = [self.W, self.hbias, self.vbias]

    def free_energy(self, v_sample):
        # this function computes the free energy
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        # this function propagates the visible units activation 
        # upwards to the hidden units
        # input is the inputs to visible units
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        # this function infers state of hidden units given visible units
        # compute the activation of the hidden units given a sample of the visibles
        # input is the sample of visible units
        
        # forward pass to get the probability of output
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default
        # if we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        # this function propagates the hidden units activation 
        # downwards to the visible units
        # input is the inputs to hidden units
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        # this function infers state of visible units given hidden units
        # compute the activation of the visible given the hidden sample
        # nput is the sample of hidden units
        
        # backward pass to get the sample (guesses about the original data) 
        # from the hidden units
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        # this function implements one step of Gibbs sampling,
        # starting from the hidden state
        # sample for hidden units
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        # this function implements one step of Gibbs sampling,
        # starting from the visible state
        # sample for visible units
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]
            
    def gibbs_vhv_clamped(self, v0_mean, v0_sample, mask):
        ###################################################################
        ###################### INSERT YOUR CODE HERE ######################
        ###################################################################
        # this function implements one step of Gibbs sampling,
        # starting from the visible state and also allows 
        # clamping for the v1_sample
        # Inputs:
        # sample for visible units and the clamping_mask to be used to mask
        # v1_sample's
        # hint 1: now the function is similar to the gibbs_vhv() function 
        # calculate v1_sample_clamped and v1_mean_clamped with the mask given
        # as input 
        # hint 2: return v1_sample_clamped and v1_mean_clamped instead of 
        # v1_sample and v1_mean
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]
        ###################################################################
        ############################ END ##################################
        ###################################################################

    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        # this functions implements one step of contrasive divergence
        # (CD-k) or persistent contrasive divergence (PCD-k)
        
        # Inputs:
        # lr: learning rate used to train the RBM

        # persistent: None for CD
        #             For PCD, shared variable containing old state 
        #            of Gibbs chain This must be a shared variable of size 
        #            (batch size, number of hidden units)
        # k: number of Gibbs steps to do in CD-k/PCD-k

        # Outputs:
        # cost 
        # updates dictionary for update rules of weights 
        # and biases as well as an update of the shared variable 
        # used to store the persistent

        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times
        # the scan will return the entire Gibbs chain
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            # the None are place holders, saying that
            # chain_start is the initial state corresponding to the
            # 6th output
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k,
            name="gibbs_hvh"
        )
        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(chain_end))
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])
        
        # constructs the update dictionary 
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(
                lr,
                dtype=theano.config.floatX
            ) 
      
        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(updates,
                                                           pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates, binarize="threshold"):
        # Stochastic approximation to the pseudo-likelihood

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        if binarize == "stochastic":
            #xi = T.round(self.input)
            xi = T.round(self.input + numpy.random.rand((self.input).shape[0],(self.input).shape[1]))
        elif binarize == "threshold":
            xi = (self.input > 0.5).astype(theano.config.floatX)
        #xi = T.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                            fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        # approximation to the reconstruction error
        cross_entropy = T.mean(
            T.sum(
                self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )

        return cross_entropy


