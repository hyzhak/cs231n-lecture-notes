import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      correct_class = y[i]
      if j == correct_class:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # ME: update dW
        # contribute to delta
        dW[:, j] += X[i]
        # in additional contribute to delta of correct class
        dW[:, correct_class] -= X[i]
        # ME: end of my update

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  dW /= num_train
  # derivative of regularization
  dW += 2 * reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X.dot(W)

  # use y as index and extract correct scores
  correct_class_scores = scores[range(num_train), y]
  margins = scores - np.expand_dims(correct_class_scores, axis=1) + 1
  # remove correct classes from margins
  margins[np.arange(y.shape[0]), y] = 0
  # only positive margins
  non_neg_margins = np.maximum(0.0, margins)
  # mean by training examples
  loss = non_neg_margins.sum() / float(num_train)
  # regularization
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  num_classes = W.shape[1]

  coeff_mat = np.zeros((num_train, num_classes))
  # take into account any class which crossed a margin
  coeff_mat[non_neg_margins > 0] = 1
  # all of incorrect classes contribute to correct class for single sample
  coeff_mat[range(num_train), y] = -np.sum(coeff_mat, axis=1)

  # current values define scale of delta
  dW = X.T.dot(coeff_mat)
  dW /= num_train 
  dW += 2 * reg * W    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
