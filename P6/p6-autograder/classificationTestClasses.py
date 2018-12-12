# classificationTestClasses.py
# ----------------------------
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


import cPickle
from hashlib import sha1
import testClasses

from collections import defaultdict
from pprint import PrettyPrinter
pp = PrettyPrinter()

from pacman import GameState
import random, math, traceback, sys, os

import torch
import dnn

import dataClassifier, samples

VERBOSE = False


# Data sets
# ---------

DIGIT_DATUM_WIDTH = 28
DIGIT_DATUM_HEIGHT = 28

def readDigitData(trainingSize=2000, testSize=1000):
    rootdata = 'digitdata/'
    # loading digits data
    testSize = 1000
    trainingData = samples.loadDataFile(rootdata + 'trainingimages', trainingSize, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile(rootdata + "traininglabels", trainingSize)
    validationData = samples.loadDataFile(rootdata + "validationimages", testSize, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile(rootdata + "validationlabels", testSize)
    testData = samples.loadDataFile("digitdata/testimages", testSize, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", testSize)
    return (trainingData, trainingLabels,
            validationData, validationLabels,
            testData, testLabels)


def readContestData(trainingSize=None, testSize=None):
    rootdata = 'pacmandata'
    trainingData, trainingLabels = samples.loadPacmanData(rootdata + '/contest_training.pkl', trainingSize)
    validationData, validationLabels = samples.loadPacmanData(rootdata + '/contest_validation.pkl', testSize)
    testData, testLabels = samples.loadPacmanData(rootdata + '/contest_test.pkl', testSize)
    return (trainingData, trainingLabels,
            validationData, validationLabels,
            testData, testLabels)


digitData = readDigitData()
contestData = readContestData()


DATASETS = {
    "digitData": lambda: digitData,
    "contestData": lambda: contestData
}

DATASETS_LEGAL_LABELS = {
    "digitData": list(range(10)),
    "contestData": ["East", 'West', 'North', 'South', 'Stop']
}


# Test classes
# ------------

def getAccuracy(data, classifier):
    trainingData, trainingLabels, validationData, validationLabels, testData, testLabels = data
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    acc = 100.0 * correct / len(testLabels)
    serialized_guesses = ", ".join([str(guesses[i]) for i in range(len(testLabels))])
    print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (acc)
    return acc, serialized_guesses


class GradeCNNTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(GradeCNNTest, self).__init__(question, testDict)

    def execute(self, grades, moduleDict, solutionDict):
        passed_all_shapes = True
        for shape, L1 in [
                (dict(C = 1, H = 11, W = 20, n_labels = 5, K1 = 3,
                      F1 = 10, P1 = 1, K2 = 4, F2 = 5, P2 = 1, L2 = 10),
                 450),
                (dict(C = 14, H = 100, W = 150, n_labels = 3, K1 = 5,
                      F1 = 10, P1 = 2, K2 = 4, F2 = 5, P2 = 3, L2 = 10),
                 1725),
                (dict(C = 9, H = 11, W = 20, n_labels = 5, K1 = 5,
                      F1 = 10, P1 = 1, K2 = 4, F2 = 5, P2 = 3, L2 = 50),
                 20),
        ]:

            self.addMessage("testing shape = %s." % (shape,))

            try:
                model = dnn.CNN(**shape)
            except Exception as e:
                self.addMessage("failed shape %s got exception %s" % (shape, e))
                passed_all_shapes = False
                continue

            if not (model.conv2.kernel_size == (shape['K2'], shape['K2'])
                    and model.conv2.padding == (0, 0)
                    and model.conv2.stride == (1, 1)):
                self.addMessage("conv2 has wrong size parameters.")
                passed_all_shapes = False

            if model.width_after_conv != L1:
                self.addMessage("wrong width after conv.")
                passed_all_shapes = False

            if not (model.fc2.in_features == L1
                    and model.fc2.out_features == shape['L2']):
                self.addMessage("fc2 has wrong size parameters.")
                passed_all_shapes = False

            for minibatch_size in [1, 15]:
                sample_tensor = torch.Tensor(minibatch_size, shape['C'], shape['H'], shape['W'])

                self.addMessage("trying to apply model to input with shape %s"
                                % (sample_tensor.shape,))

                try:
                    output = model(sample_tensor)
                except Exception as e:
                    self.addMessage("failed to take sample input got exception %s"
                                    % (e,))
                    passed_all_shapes = False
                    continue

                if not (output.shape == (minibatch_size, shape['n_labels'])):
                    self.addMessage("wrong output shape: got %s, expected %s"
                                    % (output.shape, (minibatch_size, shape['n_labels'])))
                    passed_all_shapes = False
                else:
                    self.addMessage("output shape ok")

        self.addMessage("passed all shapes: %s" % passed_all_shapes)

        return self.testPartial(grades, passed_all_shapes, 1)


class GradeClassifierTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(GradeClassifierTest, self).__init__(question, testDict)

        self.datasetName = testDict['datasetName']

        self.classifierModule = testDict['classifierModule']
        self.classifierClass = testDict['classifierClass']
        self.max_iterations = int(testDict['max_iterations']) if 'max_iterations' in testDict else None

        self.accuracyScale = int(testDict['accuracyScale'])
        self.accuracyThresholds = [int(s) for s in testDict.get('accuracyThresholds','').split()]
        self.maxPoints = len(self.accuracyThresholds) * self.accuracyScale
        self.exactOutput = testDict['exactOutput'].lower() == "true"

    def grade_classifier(self, moduleDict):

        data = DATASETS[self.datasetName]()
        legalLabels = DATASETS_LEGAL_LABELS[self.datasetName]

        classifierClass = getattr(moduleDict[self.classifierModule], self.classifierClass)

        # setting filename=None will stop us from saving the model.
        classifier = classifierClass(legalLabels, self.max_iterations, filename=None)
        return getAccuracy(data, classifier)


    def execute(self, grades, moduleDict, solutionDict):
        accuracy, guesses = self.grade_classifier(moduleDict)

        # Either grade them on the accuracy of their classifer,
        # or their exact
        if self.exactOutput:
            gold_guesses = solutionDict['guesses']
            if guesses == gold_guesses:
                totalPoints = self.maxPoints
            else:
                self.addMessage("Incorrect classification after training:")
                self.addMessage("  student classifications: " + guesses)
                self.addMessage("  correct classifications: " + gold_guesses)
                totalPoints = 0
        else:
            # Grade accuracy
            totalPoints = 0
            for threshold in self.accuracyThresholds:
                if accuracy >= threshold:
                    totalPoints += self.accuracyScale

            # Print grading schedule
            self.addMessage("%s correct (%s of %s points)" % (accuracy, totalPoints, self.maxPoints))
            self.addMessage("    Grading scheme:")
            self.addMessage("     < %s:  0 points" % (self.accuracyThresholds[0],))
            for idx, threshold in enumerate(self.accuracyThresholds):
                self.addMessage("    >= %s:  %s points" % (threshold, (idx+1)*self.accuracyScale))

        return self.testPartial(grades, totalPoints, self.maxPoints)

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)

        if self.exactOutput:
            _, guesses = self.grade_classifier(moduleDict)
            handle.write('guesses: "%s"' % (guesses,))

        handle.close()
        return True


class GradeModelTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(GradeModelTest, self).__init__(question, testDict)

        with file(testDict['load_model'], 'r') as f:
            self.model = cPickle.load(f)

        self.datasetName = testDict['datasetName']

        self.accuracyScale = int(testDict['accuracyScale'])
        self.accuracyThresholds = [int(s) for s in testDict.get('accuracyThresholds','').split()]
        self.maxPoints = len(self.accuracyThresholds) * self.accuracyScale

    def execute(self, grades, moduleDict, solutionDict):

        data = DATASETS[self.datasetName]()

        trainingData, trainingLabels, validationData, validationLabels, testData, testLabels = data

        guesses = self.model.classify(testData)
        correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
        accuracy = 100.0 * correct / len(testLabels)
        print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (accuracy)

        # Grade accuracy
        totalPoints = 0
        for threshold in self.accuracyThresholds:
            if accuracy >= threshold:
                totalPoints += self.accuracyScale

        # Print grading schedule
        self.addMessage("%s correct (%s of %s points)" % (accuracy, totalPoints, self.maxPoints))
        self.addMessage("    Grading scheme:")
        self.addMessage("     < %s:  0 points" % (self.accuracyThresholds[0],))
        for idx, threshold in enumerate(self.accuracyThresholds):
            self.addMessage("    >= %s:  %s points" % (threshold, (idx+1)*self.accuracyScale))

        return self.testPartial(grades, totalPoints, self.maxPoints)

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)
        handle.close()
        return True
