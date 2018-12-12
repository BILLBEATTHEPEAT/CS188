################################################################################
# Question 7:
#
# This question will test your understanding of the various operations involved
# in a convolutional neural network.  PLEASE READ THIS ENTIRE FILE CAREFULLY FOR
# INSTRUCTIONS.  Places in the code where you are expected to provide code are
# marked (as usual) with "YOUR CODE HERE".
#
# To run the autograder on just this question:
#
#   $ python autograder.py -q q7
#
################################################################################

import cPickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy

def YourCodeHere():
    raise NotImplementedError()


class Instance:
    """
    Abstract class that maps a "raw input" (original data point) to a tensor.
    This is a task-specific transformation.  To see an example see `DigitInstance`
    in dnn_digits.py.  Peeking ahead: In question 9, you will be asked to create
    and encoding of a pacman GameState`.
    """
    def __init__(self, original, tensor):
        self.original = original
        self.tensor = tensor


class CNN(nn.Module):
    """
    Convolutional neural network
    """
    def __init__(self,
                 # Input type: a 3-dimensional tensor with the follow dimensions
                 C,     # input number of channels
                 H,     # input height
                 W,     # input width

                 # Output type: log probability over n_labels
                 n_labels,  # number of output categories

                 # Below are the hyperparameters of our network architecture:
                 K1,    # conv kernel size (assume same for first convolutional layer)
                 F1,    # how many filters to learn in the first convolutional layer
                 P1,    # how much max pooling to do in the first convolutional layer

                 K2,    # conv kernel size (assume same for second convolutional layer)
                 F2,    # how many filters to learn in the second convolutional layer
                 P2,    # how much max pooling to do in the second convolutional layer

                 L2,    # number of hidden units in the hidden layer (between CNN output and prediction layer)
    ):
        # call base class's constructor
        super(CNN, self).__init__()

        # Fix the random seed here to make sure we get the same random
        # initialization.
        torch.manual_seed(0)

        # These variables describe the shape of our input tensor that represents a data point
        self.W = W
        self.H = H
        self.C = C

        # Hold on to our hyperparameters in case we need them later.
        self.K1 = K1
        self.P1 = P1
        self.F1 = F1

        self.K2 = K2
        self.P2 = P2
        self.F2 = F2

        self.two_convs = True
        self.hidden_layer = True

        # Here we create several pytorch layers. Each one holds the parameters
        # that will be tuned by the learning algorithm. (Other operators, such
        # as the nonlinearity and max pooling, will be applied in the `forward`
        # method because they are not stateful.)

        ########################################################################
        # PART 1:
        #
        # In the first part of question 7, we will ask you to figure out how the
        # height, width, and number of channels change change as a function of
        # the different hyperparameters given to the CNN.
        #
        # 0. We start with the height (h), width (w), and number of channels (c)
        h = self.H
        w = self.W
        c = self.C

        # The code below creates a simple convolutional layer (without any
        # pooling or nonlinearities).  You may find it helpful to review the
        # slides from class or the pytorch documentation for answering the
        # question below and/or understanding what the various operations from
        # the torch package we are using mean.  Reading *some* of the pytorch
        # documentation is expected and regarded 'part of the assignment'.
        #
        # pytorch documentation:
        #   https://pytorch.org/docs/stable/index.html
        self.conv1 = nn.Conv2d(c, F1, kernel_size=K1)       # You do not need to modify this line.

        # ***YOUR CODE HERE***
        #
        # (1.1) After applying the convolution (without padding), what is our
        # height and width?
        h = self.H - self.K1 + 1
        w = self.W - self.K1 + 1
        c = self.F1

        # (1.2) After apply max pooling what is our height and width?
        h = h / self.P1
        w = w / self.P1
        c = c

        # (1.3) After applying ReLU nonlinearity?
        h = h
        w = w
        c = c

        ########################################################################
        # PART 2: Add a second convolutional layer. Apply similar logic to part
        # 1 to infer how the shape of our tensor evolves as we pass it through
        # another conv2d layer with kernel size = K2, followed by with
        # max-pooling (P2), and a ReLU nonlinearity.
        if self.two_convs:
            # ***YOUR CODE HERE***

            # (2.1) Add a second convolutional layer with kernel size `K2`.
            self.conv2 = nn.Conv2d(c, F2, kernel_size=K2)

            # (2.2) After applying the convolution (without padding), what is our
            # height, width, and number of channels?
            h = h - self.K2 + 1
            w = w - self.K2 + 1
            c = F2

            # (2.3) After apply max pooling with `P2` what is our height, width,
            # and number of channels?
            h = h / self.P2
            w = w / self.P2
            c = c

            # (2.4) After the ReLU what is our height, width, and number of channels?
            h = h
            w = w
            c = c

            # (2.5) Now, modify the `forward` function to use `self.conv2` (see
            # the comment tagged with 2.5)

        # (2.6) how 'wide' would our data be if we flattened our 3-dimensional
        # input into a single vector?
        width = h * w * c

        # We need to remember the width at this point in the network because we
        # need to reshape our tensor from 3-dimensional to 1-dimensional.
        self.width_after_conv = width

        ########################################################################
        # PART 3: Add another fully connected layer followed by a ReLU
        # nonlinearity before our prediction layer (log softmax).
        if self.hidden_layer:
            # ***YOUR CODE HERE***

            # (3.1) Create a fully connected layer (and instance of the Linear class like the one below)
            self.fc2 = nn.Linear(self.width_after_conv, L2)

            # (3.2) How does the fully connected layer affect the width of our network?
            width = L2

            # (3.2) How does the nonlinearity affect the width of our network?
            width = width

            # (3.3) Now, modify the `forward` function to use `self.fc2` (see
            # the comment tagged with 3.3)

        self.fc1 = nn.Linear(width, n_labels)    # "fc" stands for "fully connected layer"
        self.dropout = nn.Dropout()
        ########################################################################

    # Note that the forward method is called by the call method on the model.
    # This is a behavior that is inherited by pytorch's `nn.Module` class.
    def forward(self, x):
        assert isinstance(x, torch.Tensor)

        # Apply the first convolutional layer
        x = self.conv1(x)
        x = F.max_pool2d(x, self.P1)
        x = F.relu(x)

        if self.two_convs:
            # (2.5): Add a second convolutional layer just like the one above:
            # convolutional layer with kernel size K2 -> a max pooling layer
            # with parameter P2, followed by a ReLU nonlinearity (hint: your
            # code should be *very* similar to above).
            #
            # ***YOUR CODE HERE***
            x = self.conv2(x)
            x = F.max_pool2d(x, self.P2)
            x = F.relu(x)

        # Reshape our multi-dimensional tensor into a flat tensor
        x = x.view(-1, self.width_after_conv)

        if self.hidden_layer:
            # (3.3): Add a fully connected hidden layer, dropout and a ReLU
            # nonlinearity. (Hint look below to see how to use the fully
            # connected linear layer and dropout, then look above to see how to
            # use relu).
            #
            # ***YOUR CODE HERE***
            x = self.fc2(x)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.fc1(x)
        x = self.dropout(x)              # apply dropout to this layer to improve generalization

        return F.log_softmax(x, dim=1)   # output log-probabilities of each class


# You do not need to modify this class, but do read the code.
class TrainerCNN:
    """
    Wrapper class for `CNN`, which manages training and the input/output encoding.
    """

    def __init__(self, model, actions, preprocess, learning_rate, momentum,
                 max_iterations, filename):
        self.model = model
        self.actions = actions
        self._preprocess = preprocess
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_iterations = max_iterations
        self.minibatch_size = 32
        self.filename = filename

        # Since the CNN encodes its actions as integers, we have to manage an
        # encoding from integers to the strings that they encode. So here we
        # create a bijective mapping from original action types to a unique
        # integer that will be used to encode then in the `CNN` output vector.
        self.action2index = {action: i for i, action in enumerate(actions)}
        self.index2action = {i: action for i, action in enumerate(actions)}

    def classify(self, states):
        "Make predictions about inputs (may be 'raw' inputs or preprocessed Tensors)."
        self.model.eval()   # evaluation model disables dropout so that we get an accurate prediction
        ys = []
        for s in states:
            score = self.model(self.preprocess(s))
            _, y = max((score[0,self.action2index[a]], a) for a in self.getLegalActions(s))
            ys.append(y)
        self.model.train()   # switching back to train mode re-enables dropout
        return ys

    def getLegalActions(self, x):
        if hasattr(x, 'getLegalActions'):
            return x.getLegalActions()
        else:
            return self.actions

    def preprocess(self, x):
        "Convert the input `x` into a `Tensor` if it hasn't been converted already."
        if not isinstance(x, torch.Tensor):
            if not isinstance(x, Instance):
                x = self._preprocess(x)
            if isinstance(x, Instance):
                x = x.tensor
        return x

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """Train the CNN with minibatch SGD and early stopping based on validation accuracy.

        Minibatching will make training faster because it better leverages the
        available hardware.  If we had GPUs available to use, the speed-up would
        be tremendous; on a CPU the speed-up is less, but often 4x faster than
        minibatch_size = 1.

        """
        optimizer = optim.SGD(self.model.parameters(),
                              lr = self.learning_rate * self.minibatch_size,
                              momentum = self.momentum)

        n = len(trainingData)

        print 'number of training examples:', n
        print 'number of validation examples:', len(validationData)

        # For efficiency, treprocess training and validation data up front.
        trainingData = [self._preprocess(s) for s in trainingData]
        validationData = [self._preprocess(s) for s in validationData]

        # For efficiency, we will preprocess all of the data as a big tensor.
        # This will enable fast minbatch training.
        n = len(trainingData)
        X = torch.cat([trainingData[i].tensor for i in range(n)], 0)
        Y = torch.LongTensor([self.action2index[trainingLabels[i]] for i in range(n)])

        best = (-float('inf'), {})    # used by early stopping
        self.model.train()
        for i in range(1, 1+self.max_iterations):

            print
            print 'epoch', i

            for _ in range(n // self.minibatch_size + 1):  # samples every training point approximately once.
                # sample minibatch example indices
                minibatch = torch.randint(low=0, high=n, size=(self.minibatch_size,), dtype=torch.long)
                self.model.train()
                optimizer.zero_grad()
                output = self.model(X[minibatch])
                loss = F.nll_loss(output, Y[minibatch])
                loss.backward()
                optimizer.step()

            t = self.evaluate(trainingData, trainingLabels)
            print 'train accuracy      %5.2f%%' % (100*t)
            v = self.evaluate(validationData, validationLabels)
            if v > best[0]:
                best = (v, deepcopy(self.model.state_dict()))   # early stopping

                if self.filename is not None:
                    with file(self.filename, 'w') as f:
                        cPickle.dump(self, f)
                    print 'saved model to file:', self.filename

            print 'validation accuracy %5.2f%% (best so far %5.2f%%)' % (100 * v, 100 * best[0])

        # Load the best model we say according to early stopping.
        self.model.load_state_dict(best[1])
        self.model.eval()

    def evaluate(self, x, y):
        "Compute the accuracy of the models' prediction on each of the `x[i]` compared to gold output `y[i]`."
        return np.mean([guess == truth for guess, truth in zip(self.classify(x), y)])
