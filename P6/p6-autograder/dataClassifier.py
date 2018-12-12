# dataClassifier.py
# -----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# This file contains feature extraction methods and harness
# code for data classification

import samples
import sys
import util
from pacman import GameState

import dnn_digits
import dnn_pacman

DIGIT_DATUM_WIDTH = 28
DIGIT_DATUM_HEIGHT = 28

## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def printImage(self, pixels):
        """
        Prints a Datum object that contains all pixels in the
        provided list of pixels.  This will serve as a helper function
        to the analysis function you write.

        Pixels should take the form
        [(2,2), (2, 3), ...]
        where each tuple represents a pixel.
        """
        image = samples.Datum(None,self.width,self.height)
        for pix in pixels:
            try:
            # This is so that new features that you could define which
            # which are not of the form of (x,y) will not break
            # this image printer...
                x,y = pix
                image.pixels[x][y] = 2
            except:
                print "new features:", pix
                continue
        print image

def default(str):
    return str + ' [Default: %default]'

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """


def readCommand( argv ):
    "Processes the command used to run from the command line."
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-c', '--classifier', help=default('The type of classifier'),
                      choices=['dnn_digits', 'dnn_pacman'])
    parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'pacman'])
    parser.add_option('-t', '--training', help=default('The size of the training set'), type=int)
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=15, type=int)
    parser.add_option('-s', '--validation', help=default("Amount of validation data to use"), type=int)
    parser.add_option('-g', '--agentToClone', help=default("Pacman agent to copy"), default=None)

    parser.add_option('--save', help='Filename to save model.', default=None)


    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print "Doing classification"
    print "--------------------"
    print "data:\t\t", options.data
    print "classifier:\t\t", options.classifier

    if options.data == "digits":
        legalLabels = range(10)
    else:
        legalLabels = ['Stop', 'West', 'East', 'North', 'South']

    if options.classifier == "dnn_digits":
        classifier = dnn_digits.DeepDigitClassifier(legalLabels, options.iterations, filename=options.save)

    elif options.classifier == "dnn_pacman":
        classifier = dnn_pacman.DeepPacmanClassifier(legalLabels, options.iterations, filename=options.save)

    else:
        print "Unknown classifier:", options.classifier
        print USAGE_STRING

        sys.exit(2)

    args['agentToClone'] = options.agentToClone
    args['classifier'] = classifier

    return args, options

# Dictionary containing full path to .pkl file that contains the agent's training, validation, and testing data.
MAP_AGENT_TO_PATH_OF_SAVED_GAMES = {
    'FoodAgent': ('pacmandata/food_training.pkl','pacmandata/food_validation.pkl','pacmandata/food_test.pkl' ),
    'StopAgent': ('pacmandata/stop_training.pkl','pacmandata/stop_validation.pkl','pacmandata/stop_test.pkl' ),
    'SuicideAgent': ('pacmandata/suicide_training.pkl','pacmandata/suicide_validation.pkl','pacmandata/suicide_test.pkl' ),
    'GoodReflexAgent': ('pacmandata/good_reflex_training.pkl','pacmandata/good_reflex_validation.pkl','pacmandata/good_reflex_test.pkl' ),
    'ContestAgent': ('pacmandata/contest_training.pkl','pacmandata/contest_validation.pkl', 'pacmandata/contest_test.pkl' )
}

# Main harness code
def runClassifier(args, options):
    classifier = args['classifier']

    # Load data
    if options.data == "pacman":
        agentToClone = args.get('agentToClone', None)
        trainingData, validationData, testData = MAP_AGENT_TO_PATH_OF_SAVED_GAMES.get(agentToClone, (None, None, None))
        trainingData = trainingData or args.get('trainingData', False) or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][0]
        validationData = validationData or args.get('validationData', False) or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][1]
        testData = testData or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][2]

        trainingData, trainingLabels = samples.loadPacmanData(trainingData, options.training)
        validationData, validationLabels = samples.loadPacmanData(validationData, options.validation)
        testData, testLabels = samples.loadPacmanData(testData, None)

    elif options.data == "digits":
        if options.training is None:
            options.training = 2000
        if options.validation is None:
            options.validation = 1000
        numTest = 1000

        trainingData = samples.loadDataFile("digitdata/trainingimages", options.training, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", options.training)

        validationData = samples.loadDataFile("digitdata/validationimages", options.validation, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("digitdata/validationlabels", options.validation)

        testData = samples.loadDataFile("digitdata/testimages", numTest, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)

    else:
        raise ValueError('unrecognized dataset %r' % options.data)

    # Conduct training and testing
    print "Training..."
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    print "Testing..."
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))


if __name__ == '__main__':
    # Read input
    args, options = readCommand(sys.argv[1:])
    # Run classifier
    runClassifier(args, options)
