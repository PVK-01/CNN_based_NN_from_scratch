import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from utils.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #
    c, h , w = input_dim
    pad = (filter_size - 1) / 2
    self.params['W1'] = np.random.randn(num_filters, c, filter_size, filter_size) * weight_scale
    self.params['b1'] = np.zeros(num_filters)
    conv_out_h = int(1 + 2 *pad + h - filter_size)
    pool_out_h = int(conv_out_h / 2)
    conv_out_w = int(1 +  2*pad + w - filter_size)
    pool_out_w = int(conv_out_w / 2)
    self.params['W2'] = np.random.randn((num_filters * pool_out_w * pool_out_h), hidden_dim) * weight_scale
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale
    self.params['b3'] = np.zeros(num_classes)
    self.params['gamma2'] = np.ones(hidden_dim)
    self.params['beta2'] = np.ones(hidden_dim)
    if self.use_batchnorm:
      self.params['gamma1'] = np.zeros(num_filters)
      self.params['beta1'] = np.zeros(num_filters)
      
    
    

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
    self.bn_param = {}
    if self.use_batchnorm:
      self.bn_param['mode'] = 'train'
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #
    mode = 'test' if y is None else 'train'
    bn_param = self.bn_param
    bn_param['mode'] = mode
    gamma2, beta2 = self.params['gamma2'], self.params['beta2']
    if self.use_batchnorm:
      gamma1, beta1 = self.params['gamma1'], self.params['beta1']
      out_1, cache_1 = conv_batchnorm_relu_pool_forward(X, W1, b1, gamma1, beta1, conv_param, pool_param, bn_param)
      out_2, cache_2 = affine_batchnorm_relu_forward(out_1, W2, b2, gamma2, beta2 , self.bn_param)
      scores, cache_3 = affine_forward(out_2, W3, b3)
    else:
      out_1, cache_1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
      out_2, cache_2 = affine_batchnorm_relu_forward(out_1, W2, b2, gamma2, beta2 , self.bn_param)
      scores, cache_3 = affine_forward(out_2, W3, b3)
    
    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    if y is None:
      return scores
    
    loss, grads = 0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #
    s_loss, s_grad = softmax_loss(scores, y)
    loss = 0.5 * self.reg * np.sum(np.square(W1)) + 0.5 * self.reg * np.sum(np.square(W2)) +0.5 * self.reg * np.sum(np.square(W3)) + s_loss
    dx_3, grads['W3'], grads['b3']  = affine_backward(s_grad, cache_3)
    dx_2, grads['W2'], grads['b2'], grads['gamma2'], grads['beta2'] = affine_batchnorm_relu_backward(dx_3, cache_2)
    if self.use_batchnorm:
      dx_1, grads['W1'], grads['b1'], grads['gamma1'], grads['beta1'] = conv_batchnorm_relu_pool_backward(dx_2, cache_1)
    else:  
      dx_1, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx_2, cache_1)

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads
  
  
pass
