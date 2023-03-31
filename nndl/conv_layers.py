import numpy as np
from nndl.layers import *
import pdb


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # =============================================================== #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  
  hw, ww = w.shape[2], w.shape[3]
  h1 =  int(1 + ( x.shape[2] + 2 * pad - hw) / stride)
  b1 = int(1 + ( x.shape[3] + 2 * pad - ww) / stride)
  out = np.zeros([int(x.shape[0]), int(w.shape[0]), h1, b1])
  p = ((0,0), (0,0), (pad,pad), (pad,pad))
  x_padded = np.pad(x, pad_width = p, mode='constant', constant_values=0)
  
  for n in range(x.shape[0]):
      for f in range(w.shape[0]):
        for i in np.arange(h1):
            for j in np.arange(b1):
                out[n, f, i, j] = np.sum(w[f,:,:,:] * x_padded[n, :, i*stride:i*stride+hw, j*stride:j*stride+ww]) + b[f]
              

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  db = np.zeros(b.shape)
  dw = np.zeros_like(w)
  dx_padded = np.zeros_like(xpad)
  for i in range(N):
      for j in range(F):
          for h_out in range(out_height):
              for w_out in range(out_width):
                  dw[j] = dw[j] + xpad[i, :,  (h_out*stride) : ((h_out*stride) + f_height), (w_out*stride) : ((w_out*stride) + f_width)] * dout[i,j,h_out,w_out]
                  dx_padded[i, :,  (h_out*stride) : ((h_out*stride) + f_height), (w_out*stride) : ((w_out*stride) + f_width)] += w[j] * dout[i,j,h_out,w_out]
                  db[j] = db[j] + dout[i, j, h_out, w_out]
                  dx = dx_padded[:, :, pad : -pad, pad : -pad]
  

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  p_h, p_w, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  h_out = int(1 + ( x.shape[2] - p_h) / stride)
  w_out = int(1 + ( x.shape[3] - p_w) / stride)
  out = np.zeros([x.shape[0], x.shape[1], h_out, w_out])
  for i in range(x.shape[0]):
      for c in range(x.shape[1]):
          for h1 in range(h_out):
              for w1 in range(w_out):
                  out[i, c, h1, w1] = np.max(x[i, c, h1 * stride : h1 * stride + p_h, w1 * stride : w1 * stride + p_w])

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  p_h, p_w, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  h_out = int(1 + ( x.shape[2] - p_h) / stride)
  w_out = int(1 + ( x.shape[3] - p_w) / stride)
  dx = np.zeros_like(x)
  for i in range(x.shape[0]):
      for c in range(x.shape[1]):
          for h1 in range(h_out):
              for w1 in range(w_out):
                  max_ind = np.unravel_index(np.argmax(x[i, c, h1 * stride : h1 * stride + p_h, w1 * stride : w1 * stride + p_w]), (p_h, p_w))
                  dx[i, c, h1 * stride + max_ind[0], w1 * stride + max_ind[1]] = dout[i, c, h1, w1]
  pass

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  n, c, h, w = x.shape
  x = x.transpose((0,2,3,1))
  x = x.reshape((n*w*h, c))
  out, cache = batchnorm_forward(x, gamma, beta, bn_param)
  out = out.reshape((n, h, w, c))
  out = out.transpose((0, 3, 1, 2))
 

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  n, c, h, w = dout.shape
  dout = dout.transpose((0,2,3,1))
  dout = dout.reshape((n*w*h, c))
  dx, dgamma, dbeta = batchnorm_backward(dout, cache)
  dx = dx.reshape((n, h, w, c))
  dx = dx.transpose((0, 3, 1, 2))  

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta