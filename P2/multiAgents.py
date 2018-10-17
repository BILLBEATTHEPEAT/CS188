# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()

        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newCapsules = currentGameState.getCapsules()

        score = (currentGameState.getNumFood() - successorGameState.getNumFood())*100

        cls_food_dist = 99999
        cls_ghost_dist = 99999

        if successorGameState.isWin():
            return 100000

        newFood = newFood.asList()

        if newPos in newCapsules:
            score += 100

        for pos in newFood:
            dis = abs(pos[0] - newPos[0]) + abs(pos[1] - newPos[1])
            if dis < cls_food_dist:
                cls_food_dist = dis

        for i in range(len(newGhostStates)):
            pos = newGhostStates[i].getPosition()
            dis = abs(pos[0] - newPos[0]) + abs(pos[1] - newPos[1])
            if dis < cls_ghost_dist:
                cls_ghost_dist = dis
                safe_dis = (newScaredTimes[i] - dis)

        if safe_dis > -2 and safe_dis < 1:
            return -9999
        return score - 5*cls_food_dist


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        actions = gameState.getLegalActions(0)
        max_score = -99999
        best_action = actions[0]
        # print 0, actions
        for action in actions:
            successorState = gameState.generateSuccessor(0, action)
            if successorState.isWin():
                return action
            score = self.get_best_value(successorState, self.depth, 1)
            if score > max_score:
                max_score = score
                best_action = action
        return best_action

    def get_best_value(self, gameState, depth, agentID):
        numAgents = gameState.getNumAgents()
        agentID = agentID % (numAgents)
        if agentID == 0:
            depth -= 1
        if depth == 0:
            return self.evaluationFunction(gameState)
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        elif agentID == 0:
            actions = gameState.getLegalActions(0)
            max_score = -99999
            # best_action = actions[0]
            for action in actions:
                successorState = gameState.generateSuccessor(0, action)
                # if successorState.isWin():
                #     return action
                score = self.get_best_value(successorState, depth, 1)
                if score > max_score:
                    max_score = score
                        # best_action = action

            # print depth, agentID, 0, max_score, actions
            return max_score

        else:
            actions = gameState.getLegalActions(agentID)
            min_score = 99999
            # best_action = actions[0]
            for action in actions:
                successorState = gameState.generateSuccessor(agentID, action)
                # if successorState.isLose():
                #     return action
                score = self.get_best_value(successorState, depth, agentID+1)
                if score < min_score:
                    min_score = score
                    # best_action = action
            # print depth, agentID, min_score, actions
            return min_score



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        actions = gameState.getLegalActions(0)
        max_score = -99999
        best_action = actions[0]
        # alpha = -999999
        # beta = 999999
        alpha, beta = -999999, 999999
        for action in actions:
            successorState = gameState.generateSuccessor(0, action)
            if successorState.isWin():
                return action
            score = self.get_best_value(successorState, self.depth, 1, alpha, beta)
            if score > max_score:
                max_score = score
                best_action = action
            alpha = max(alpha, max_score)
        return best_action

    def get_best_value(self, gameState, depth, agentID, alpha, beta):
        numAgents = gameState.getNumAgents()
        agentID = agentID % (numAgents)
        if agentID == 0:
            depth -= 1
        if depth == 0:
            return self.evaluationFunction(gameState)
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        elif agentID == 0:
            actions = gameState.getLegalActions(0)
            max_score = -99999
            # best_action = actions[0]
            for action in actions:
                successorState = gameState.generateSuccessor(0, action)
                # if successorState.isWin():
                #     return action
                score = self.get_best_value(successorState, depth, 1, alpha, beta)
                if score > max_score:
                    max_score = score
                if max_score > beta:
                    break
                alpha = max(alpha, max_score)
            return max_score

        else:
            actions = gameState.getLegalActions(agentID)
            min_score = 99999
            # best_action = actions[0]
            for action in actions:
                successorState = gameState.generateSuccessor(agentID, action)
                # if successorState.isLose():
                #     return action
                score = self.get_best_value(successorState, depth, agentID+1, alpha, beta)
                if score < min_score:
                    min_score = score
                if min_score < alpha:
                    return min_score

                beta = min(beta, min_score)
            # print depth, agentID, min_score, alpha_beta
            return min_score



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        actions = gameState.getLegalActions(0)
        max_score = -99999
        best_action = actions[0]
        # print 0, actions
        for action in actions:
            successorState = gameState.generateSuccessor(0, action)
            if successorState.isWin():
                return action
            score = self.get_best_value(successorState, self.depth, 1)
            if score > max_score:
                max_score = score
                best_action = action
        return best_action

    def get_best_value(self, gameState, depth, agentID):
        numAgents = gameState.getNumAgents()
        agentID = agentID % (numAgents)
        if agentID == 0:
            depth -= 1
        if depth == 0:
            return self.evaluationFunction(gameState)
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        elif agentID == 0:
            actions = gameState.getLegalActions(0)
            max_score = -99999
            # best_action = actions[0]
            for action in actions:
                successorState = gameState.generateSuccessor(0, action)
                # if successorState.isWin():
                #     return action
                score = self.get_best_value(successorState, depth, 1)
                if score > max_score:
                    max_score = score
                        # best_action = action

            # print depth, agentID, 0, max_score, actions
            return max_score

        else:
            actions = gameState.getLegalActions(agentID)
            expect_score = 0
            # best_action = actions[0]
            for action in actions:
                successorState = gameState.generateSuccessor(agentID, action)
                # if successorState.isLose():
                #     return action
                score = self.get_best_value(successorState, depth, agentID+1)
                expect_score += score
                    # best_action = action
            # print depth, agentID, min_score, actions
            return expect_score * 1.0 / len(actions)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    if currentGameState.isWin():
        return 9999999
    elif currentGameState.isLose():
        return -9999999
    currentPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newCapsules = currentGameState.getCapsules()
    score = scoreEvaluationFunction(currentGameState)
    score += (currentGameState.getNumFood() + len(newCapsules)) * -10

    cls_food_dist = 99999
    cls_ghost_dist = 99999
    far_food_dist = 0


    newFood = newFood.asList()

    for pos in newFood:
        dis = abs(pos[0] - currentPos[0]) + abs(pos[1] - currentPos[1])
        if dis < cls_food_dist:
            cls_food_dist = dis
        cls_food_dist = min(cls_food_dist, dis)
        far_food_dist = max(far_food_dist, dis)

    for i in range(len(newGhostStates)):
        pos = newGhostStates[i].getPosition()
        dis = abs(pos[0] - currentPos[0]) + abs(pos[1] - currentPos[1])
        if dis < cls_ghost_dist:
            cls_ghost_dist = dis
            safe_dis = (newScaredTimes[i] - dis)

    if safe_dis > -2 and safe_dis < 1:
        return -99999
    score = score - 2 * cls_food_dist - far_food_dist
    return score

# Abbreviation
better = betterEvaluationFunction

