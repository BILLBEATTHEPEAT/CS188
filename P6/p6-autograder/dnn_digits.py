################################################################################
# Question 8:
#
# In this question, we will apply the convolutional neural network from Question
# 7 to the digit classification problem.  Again: PLEASE READ THIS ENTIRE FILE
# CAREFULLY FOR INSTRUCTIONS.  Places in the code where you are expected to
# provide code are marked (as usual) with "YOUR CODE HERE".  To test your
# solution run:
#
#   $ python dataClassifier.py -c dnn_digits -d digits --save digits.model
#
# Note the option `--save digits.model`.  This option will save your trained
# model to a file named `digits.model`.  The autograder will evaluate the
# performance of this saved model, but it will ALSO try training your model from
# scratch (using the same fixed random seed).  Make sure to include
# `digits.model` in your submission!
#
# To run the autograder on just this question:
#
#   $ python autograder.py -q q8
#
################################################################################

import torch
from dnn import Instance, CNN, TrainerCNN
from samples import Datum

def YourCodeHere():
    raise NotImplementedError()


# You do not need to modify this class, but do read the code.
class DigitInstance(Instance):
    "Convert Berkeley's `dict`-based image format to a pytorch-friendly 4-dimensional `Tensor`."

    def __init__(self, original):
        assert isinstance(original, Datum), original
        # The original `datum` is a data structure that represents a 28 x 28
        # image with only 1 color channel (grayscale).  Here we will convert
        # from that representation to a pytorch-friendly representation.
        # Specifically, a Tensor object with the following shape:
        #
        #   (minibatch index, input color channel, pixel row, pixel column)
        #
        # To read more about the dimensions of this tensor (e.g., what's the
        # deal with the minibatch index), you may look at the class
        # `PacmanInstance` in dnn_pacman.py.
        tensor = torch.Tensor(1, 1, 28, 28)

        for x in range(28):
            for y in range(28):
                tensor[0,0,y,x] = original.getPixel(x, y)

        # call the initializer for the base class.
        Instance.__init__(self, original, tensor)

    def getLegalActions(self):
        return list(range(10))


class DeepDigitClassifier(TrainerCNN):
    def __init__(self, actions, max_iterations, filename):

        ########################################################################
        # Part 1:
        #
        # Configure the convolutional network by setting the following
        # hyperparameters in the constructor to the CNN class and well as the
        # parameters of the training algorithm.  Finding a good setting of these
        # hyperparameters will require some trial and error.  To get full
        # credit, you should achieve a test score of >=90% on the digit dataset
        # (half points for >=85%).
        #
        # *** YOUR CODE GOES HERE ***

        model = CNN(

            # Input shape
            H  = 28,   # input height
            W  = 28,   # input width
            C  = 1,    # input number of channels

            # Output shape
            n_labels = len(actions),  # number of output categories

            # First convolutional layer:
            K1 = 5,    # conv kernel size (assume same for both layers)
            P1 = 2,    # how much max pooling to do in the first convolutional layer
            F1 = 10,    # how many filters to learn in the first convolutional layer

            # Second convolutional layer:
            K2 = 5,    # conv kernel size (assume same for both layers)
            P2 = 2,    # how much max pooling to do in the second convolutional layer
            F2 = 20,   # how many filters to learn in the second convolutional layer

            # Hidden layer before the prediction layer:
            L2 = 100,    # number of hidden units in the hidden layer (between CNN output and prediction layer)
        )

        TrainerCNN.__init__(self,
                            model,
                            actions = actions,           # all possible output categories
                            preprocess = DigitInstance,  # map a "raw input" (e.g., GameState) to a pytorch-CNN-friendly Tensor object
                            learning_rate = 0.005,
                            momentum = 0.5,
                            max_iterations = max_iterations,
                            filename = filename)
