import numpy as np


def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        ### YOUR CODE HERE
        x = x - np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(x) 
        x = e_x / np.sum(e_x, axis=-1, keepdims=True)
        ### END YOUR CODE
    else:
        # Vector
        ### YOUR CODE HERE
        x = x - np.max(x)
        e_x = np.exp(x)
        x = e_x / np.sum(e_x)
        ### END YOUR CODE

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    test1 = softmax(np.array([1, 2]))
    print(test1)
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001, 1002], [3, 4]]))
    print(test2)
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001, -1002]]))
    print(test3)
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print("You should be able to verify these results by hand!\n")


def your_softmax_test():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print("Running your tests...")
    ### YOUR OPTIONAL CODE HERE
    # Test 1: Test with zeros array
    test1 = softmax(np.zeros(3))
    ans1 = np.array([1/3, 1/3, 1/3])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)
    print("Test 1 passed: Uniform distribution for zero inputs")

    # Test 2: Test with very large negative numbers
    test2 = softmax(np.array([-1000, -1200, -1400]))
    print(test2)
    assert np.sum(test2) == 1
    assert test2[0] > test2[1] > test2[2]
    print("Test 2 passed: Proper handling of large negative numbers")

    # Test 3: Test with 3D array
    test3 = softmax(np.array([[[1,2], [3,4]], [[5,6], [7,8]]]))
    assert test3.shape == (2,2,2)
    assert np.allclose(np.sum(test3, axis=-1), 1)
    print("Test 3 passed: Works with 3D arrays")

    # Test 4: Test with single number
    test4 = softmax(np.array([1.0]))
    assert np.allclose(test4, [1.0])
    print("Test 4 passed: Works with single number")

    # Test 5: Test probability sum
    test5 = softmax(np.array([1, 2, 3, 4, 5]))
    assert np.isclose(np.sum(test5), 1)
    assert np.all(test5 > 0)
    print("Test 5 passed: Sum of probabilities equals 1 and all values positive")
    ### END YOUR CODE


if __name__ == "__main__":
    test_softmax_basic()
    your_softmax_test()
