################################################################################
# Question 9
#
# In this question, we will apply the convolutional neural network from Question
# 7 to the pacman behavioral cloning dataset.  Again: PLEASE READ THIS ENTIRE
# FILE CAREFULLY FOR INSTRUCTIONS.  Places in the code where you are expected to
# provide code are marked (as usual) with "YOUR CODE HERE".  To test your
# solution run:
#
# To develop your solution run:
#   $ python dataClassifier.py -c dnn_pacman -d pacman -g ContestAgent --save pacman.model
#
# Note the option `--save pacman.model`.  This option will save your trained
# model to a file named `pacman.model`.  The autograder will evaluate the
# performance of this saved model, but it will ALSO try training your model from
# scratch (using the same fixed random seed).  Make sure to include
# `pacman.model` in your submission!
#
# To run the autograder use:
#   $ python autograder.py -q q9
#
################################################################################

import torch
from pacman import GameState
from dnn import Instance, CNN, TrainerCNN


def YourCodeHere():
    raise NotImplementedError()


class PacmanInstance(Instance):
    def __init__(self, state):
        assert isinstance(state, GameState), state

        ########################################################################
        # PART 1
        #
        # Implement a tensor encoding of the pacman GameState:
        #
        # We are going to use the string-based representation of the GameState
        # object to encode the pacman state as a pytorch friendly Tensor object.
        #
        #     >>> print str(state)
        #     %%%%%%%%%%%%%%%%%%%%
        #     %    %     G  %....%
        #     % %% % %%%%%% %.%%.%
        #     % %.          ..G%.%
        #     % %.%% %%  %%.%%.%.%
        #     %  >   %    %......%
        #     %.% %%.%%%%%%.%%.%.%
        #     %.%    ..........%.%
        #     %.%%.% %%%%%%.%.%%.%
        #     %....%    ....%...o%
        #     %%%%%%%%%%%%%%%%%%%%
        #     Score: 580
        #
        # As you can see we get a 2d grid of the following 9 symbols,
        #
        #    [' ', '%', 'G', 'o', '.', 'v', '^', '<', '>']
        #
        # Ok, technically we didn't see ['v', '^', '<'], but trust me you will!
        #
        # Here is a glossary for what these symbols mean:
        #
        #   ['v', '^', '<', '>'] are tell us pacman's current position and orientation.
        #   '%' is a wall
        #   ' ' is an empty space (not a wall)
        #   '.' is food
        #   'G' is a ghost
        #   'o' is a scared ghost

        s = str(state).split('\n')[:-2]         # list of strings (drop last 2 lines because they correspond to the string encoding of current score).
        H = len(s)                              # height
        W = len(s[0])                           # width

        # You may assume that all GameStates are H=11, W=20
        assert H == 11 and W == 20

        # Here, you will have free reign to encode the `GameState1 however you
        # like.  The only catch is in order to get full credit for the
        # assignment your code needs to get a test score of >= 85% (as test
        # score of >= 80% gets half credit).
        #
        # We highly recommend exploring a few tensor-encoding strategies, but to
        # get your creative juices flowing, here are some initial idea (again,
        # you are free to ignore all of these suggestions):
        #
        #  * Idea 1: Treat each of the 9 unique symbols that may appear in the
        #    grid position `(y,x)` as an binary encoding of one of 9 "color
        #    channels" at position `(y,x)`.
        #
        #  * Idea 2: Merge the different symbols into fewer input channels,
        #    possibly with a non-binary encoding (e.g., '%' -> -1, ' ' -> +1)
        #
        # *** YOUR CODE GOES HERE ***

        # Declare your tensor's dimensions locally.  But as a heads up!  You
        # should make sure that `DeepPacmanClassifier` also knows the correct
        # size of your input tensor!
        my_width = W
        my_height = H
        my_channels = 9

        # Technical note: The first dimension of the tensor is used for
        # minibatching.  We have handled minibatching for you in this assignment
        # (see TrainerCNN).  However, pytorch's builtin operations assume
        # minibatch -- so we will simply pass in a minibatch of size one.
        #
        # The tensor is initially all zeros. You can values into it as follows:
        #
        #    tensor[0, channel, row, column] = value
        #
        # Make sure to leave the zero in the first position.
        #
        # For another example of how to use the tensor, have a look at
        # `DigitInstance` in dnn_digits.py.
        #
        tensor = torch.zeros((1, my_channels, my_width, my_height))

        # YourCodeHere()   # extra hint: use the list of strings `s` (defined above)
        di = {' ':0, '%':1, 'G':2, 'o':3, '.':4, 'v':5, '^':6, '<':7, '>':8}
        # print s
        for x in xrange(H):
            for y in xrange(W):
                c = s[x][y]
                tensor[0, di[c], y, x] = 1

        Instance.__init__(self, state, tensor)

    def getLegalActions(self):
        return self.original.getLegalActions()


class DeepPacmanClassifier(TrainerCNN):
    def __init__(self, legalLabels, max_iterations, filename):

        ########################################################################
        # PART 2
        #
        # Just like in Q8, configure the training algorithm and CNN model
        # hyperparameters.  Note that the best settings pacman may differ from
        # digit classification.
        #
        # *** YOUR CODE HERE ***

        model = CNN(
            n_labels = len(legalLabels),   # number of output categories
            H = 11,     # input height
            W = 20,     # input width
            C = 9,     # input channels

            K1 = 3,    # conv kernel size (assume same for first layer)
            P1 = 2,    # pooling
            F1 = 10,    # number of conv filters learned in first layer

            K2 = 3,    # conv kernel size (assume same for second layer)
            P2 = 2,    # pooling
            F2 = 20,    # number of conv filters learned in second layer

            L2 = 100,    # hidden layer size
        )
        TrainerCNN.__init__(self,
                            model,
                            actions = legalLabels,
                            preprocess = PacmanInstance,  # map a "raw input" (e.g., GameState) to a pytorch-CNN-friendly Tensor object
                            learning_rate = 0.005,
                            momentum = 0.5,
                            max_iterations = max_iterations,
                            filename = filename)
