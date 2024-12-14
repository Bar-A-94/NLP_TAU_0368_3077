#!/usr/bin/env python
import random

import numpy as np

from helpers.utils import normalize_rows, sigmoid, get_negative_samples
from q2a_softmax import softmax
from q2b_gradcheck import gradcheck_naive


def naive_softmax_loss_and_gradient(
        center_word_vec,
        outside_word_idx,
        outside_vectors,
        dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models.

    Arguments:
    center_word_vec -- numpy ndarray, center word's embedding
                    (v_c in the pdf handout)
    outside_word_idx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outside_vectors -- outside vectors (rows of matrix) for all words in vocab
                      (U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    grad_center_vec -- the gradient with respect to the center word vector
                     (dJ / dv_c in the pdf handout)
    grad_outside_vecs -- the gradient with respect to all the outside word vectors
                    (dJ / dU)
    """

    ### YOUR CODE HERE

    # The predicted distribution y_hat is the probability distribution P (O|C = c) given by our model in equation 1 from pdf
    y_hat = softmax(outside_vectors @ center_word_vec)  # (v, d) @ (d, 1) = (v, 1)
    loss = -np.log(y_hat[outside_word_idx])  # scalar

    # y is the one-hot encoding of the outside word index
    y = np.zeros(y_hat.shape)  # (v,1)
    y[outside_word_idx] = 1

    # From written derivation
    grad_outside_vecs = np.outer(y_hat - y, center_word_vec)  # (v, 1) (d, 1) = (v , d)
    grad_center_vec = outside_vectors.T @ (y_hat - y)  # (d, v) @ (v, 1) = (d, 1)

    ### END YOUR CODE

    return loss, grad_center_vec, grad_outside_vecs


def neg_sampling_loss_and_gradient(
        center_word_vec, 
        outside_word_idx,
        outside_vectors,
        dataset,
        K=10
):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a center_word_vec
    and a outside_word_idx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naive_softmax_loss_and_gradient
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    neg_sample_word_indices = get_negative_samples(outside_word_idx, dataset, K)
    indices = [outside_word_idx] + neg_sample_word_indices

    ### YOUR CODE HERE

    # Pre-compute dot products and sigmoid values
    uv = outside_vectors @ center_word_vec  # (v, d) @ (d, 1) = (v, 1)
    sig_uv = sigmoid(uv)  # (v, 1)
    sig_minus_uv = 1 - sig_uv  # (v, 1)
    # Loss
    loss = -np.log(sig_uv[outside_word_idx]) - np.sum(np.log(sig_minus_uv[neg_sample_word_indices]))  # scalar
    # Grad center vector
    grad_center_positive = -(1-sig_uv[outside_word_idx])*outside_vectors[outside_word_idx]
    grad_center_negative = (1-sig_minus_uv[neg_sample_word_indices]).T @ outside_vectors[neg_sample_word_indices]  #(1, k) @ (k, d) = (1, d)
    grad_center_vec = grad_center_positive + grad_center_negative 
    # Grad outside vector
    grad_outside_vecs = np.zeros(outside_vectors.shape)
    # Positive samples
    grad_outside_vecs[outside_word_idx] = -(1-sig_uv[outside_word_idx])*center_word_vec  # (d, 1) (d, 1) = (d, 1)
    # Handle multiple occurrences of negative samples
    unique, idx, counts = np.unique(neg_sample_word_indices, return_index=True, return_counts=True)
    sig_unique_samples = sig_minus_uv[neg_sample_word_indices][idx]
    grad_outside_vecs[unique, :] = ((1 - sig_unique_samples) * counts)[:, None] * center_word_vec[None, :]

    ### END YOUR CODE

    return loss, grad_center_vec, grad_outside_vecs


def skipgram(current_center_word, outside_words, word2ind,
             center_word_vectors, outside_vectors, dataset,
             word2vec_loss_and_gradient=naive_softmax_loss_and_gradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    current_center_word -- a string of the current center word
    window_size -- integer, context window size
    outside_words -- list of no more than 2*window_size strings, the outside words
    word2ind -- a dictionary that maps words to their indices in
              the word vector list
    center_word_vectors -- center word vectors (as rows) for all words in vocab
                        (V in pdf handout)
    outside_vectors -- outside word vectors (as rows) for all words in vocab
                    (U in pdf handout)
    word2vec_loss_and_gradient -- the loss and gradient function for
                               a prediction vector given the outside_word_idx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    grad_center_vecs -- the gradient with respect to the center word vectors
            (dJ / dV in the pdf handout)
    grad_outside_vectors -- the gradient with respect to the outside word vectors
                        (dJ / dU in the pdf handout)
    """
    loss = 0.0
    grad_center_vecs = np.zeros(center_word_vectors.shape)
    grad_outside_vectors = np.zeros(outside_vectors.shape)

    ### YOUR CODE HERE

    center_word_idx = word2ind[current_center_word]
    center_word_vec = center_word_vectors[center_word_idx]
    for outside_word in outside_words:
        outside_word_idx = word2ind[outside_word]
        loss_out, grad_center_vecs_out, grad_outside_vectors_out = word2vec_loss_and_gradient(center_word_vec,
                                                                                outside_word_idx,
                                                                                outside_vectors,
                                                                                dataset)
        loss += loss_out
        grad_center_vecs[center_word_idx] += grad_center_vecs_out
        grad_outside_vectors += grad_outside_vectors_out
    ### END YOUR CODE

    return loss, grad_center_vecs, grad_outside_vectors


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################


def word2vec_sgd_wrapper(word2vec_model, word2ind, word_vectors, dataset,
                         window_size, word2vec_loss_and_gradient=naive_softmax_loss_and_gradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(word_vectors.shape)
    N = word_vectors.shape[0]
    center_word_vectors = word_vectors[:int(N / 2), :]
    outside_vectors = word_vectors[int(N / 2):, :]
    for i in range(batchsize):
        window_size1 = random.randint(1, window_size)
        center_word, context = dataset.getRandomContext(window_size1)

        c, gin, gout = word2vec_model(
            center_word, context, word2ind, center_word_vectors,
            outside_vectors, dataset, word2vec_loss_and_gradient
        )
        loss += c / batchsize
        grad[:int(N / 2), :] += gin / batchsize
        grad[int(N / 2):, :] += gout / batchsize

    return loss, grad


def test_word2vec_basic():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    dataset = type('dummy', (), {})()

    def dummy_sample_token_idx():
        return random.randint(0, 4)

    def get_random_context(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], \
            [tokens[random.randint(0, 4)] for _ in range(2*C)]
    dataset.sampleTokenIdx = dummy_sample_token_idx
    dataset.getRandomContext = get_random_context

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalize_rows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])

    print("==== Gradient check for skip-gram with naive_softmax_loss_and_gradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, naive_softmax_loss_and_gradient),
        dummy_vectors, "naive_softmax_loss_and_gradient Gradient")

    print("==== Gradient check for skip-gram with neg_sampling_loss_and_gradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, neg_sampling_loss_and_gradient),
                    dummy_vectors, "neg_sampling_loss_and_gradient Gradient")

    print("\n=== Results ===")
    print("Skip-Gram with naive_softmax_loss_and_gradient")

    print("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
            *skipgram("c", ["a", "b", "e", "d", "b", "c"], dummy_tokens,
                      dummy_vectors[:5, :], dummy_vectors[5:, :], dataset)
        )
    )

    print("Expected Result: Value should approximate these:")
    print("""Loss: 11.16610900153398
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-1.26947339 -1.36873189  2.45158957]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
Gradient wrt Outside Vectors (dJ/dU):
 [[-0.41045956  0.18834851  1.43272264]
 [ 0.38202831 -0.17530219 -1.33348241]
 [ 0.07009355 -0.03216399 -0.24466386]
 [ 0.09472154 -0.04346509 -0.33062865]
 [-0.13638384  0.06258276  0.47605228]]
    """)

    print("Skip-Gram with neg_sampling_loss_and_gradient")
    print("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\n Gradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
        *skipgram("c", ["a", "b"], dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :],
                  dataset, neg_sampling_loss_and_gradient)
        )
    )
    print("Expected Result: Value should approximate these:")
    print("""Loss: 16.15119285363322
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-4.54650789 -1.85942252  0.76397441]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
 Gradient wrt Outside Vectors (dJ/dU):
 [[-0.69148188  0.31730185  2.41364029]
 [-0.22716495  0.10423969  0.79292674]
 [-0.45528438  0.20891737  1.58918512]
 [-0.31602611  0.14501561  1.10309954]
 [-0.80620296  0.36994417  2.81407799]]
    """)


if __name__ == "__main__":
    test_word2vec_basic()
