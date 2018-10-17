# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    state = problem.getStartState()

    state_record = util.Stack()
    step_record = util.Stack()
    # step_record = []
    plan = []
    visit = []

    print "Start:", state

    if problem.isGoalState(state):
        return plan

    # state_record.push(state)
    # while state_record.isEmpty() == 0:
    #
    #     state = state_record.pop()
    #     successors = problem.getSuccessors(state)
    #     temp = successors[:]
    #     for i in range(len(successors)):
    #         if (successors[i][0]) in visit:
    #             temp.remove(successors[i])
    #     successors = temp
    #     if len(successors) == 0:
    #         step_record = step_record[:-1]
    #         continue
    #     else:
    #         successor = successors[0]
    #
    #     state_record.push(state)
    #     state_record.push(successor[0])
    #     step_record += [successor[1]]
    #     visit.append(successor[0])
    #
    #     if problem.isGoalState(successor[0]):
    #         break
    # print plan
    # return step_record


    state_record.push(state)
    successors = problem.getSuccessors(state)
    for i in range(len(successors)):
        if (successors[i][0]) not in visit:
            state_record.push(successors[i][0])
            step_record.push(successors[i][1])

    while state_record.isEmpty() == 0:

        state = state_record.pop()

        if problem.isGoalState(state):
            state_record.push(state)
            step = step_record.pop()
            plan += [step]
            break
        elif state in visit:
            step_record.pop()
            plan = plan[:-1]
            visit.remove(state)
            continue
        else:
            state_record.push(state)
            step = step_record.pop()
            plan += [step]
            visit.append(state)
            step_record.push(step)

        successors = problem.getSuccessors(state)
        # if len(successors) == 1:
        #     if not problem.isGoalState(successors[0][0]):
        #         continue
        for i in range(len(successors)):
            if (successors[i][0]) not in visit:
                state_record.push(successors[i][0])
                step_record.push(successors[i][1])

    return plan

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    state = problem.getStartState()

    state_record = util.Queue()
    step_record = util.Queue()
    # step_record = []
    plan = []
    visit = []

    print "Start:", state

    if problem.isGoalState(state):
        return []

    state_record.push(state)
    step_record.push(plan)
    # successors = problem.getSuccessors(state)
    # for i in range(len(successors)):
    #     state_record.push(successors)
    #     step_record.push([successors[i][1]])

    while state_record.isEmpty() == 0:

        state = state_record.pop()
        plan = step_record.pop()
        if problem.isGoalState(state):
            # state_record.push(state)
            # step = step_record.pop()
            # plan += [step]
            # plan = step_record.pop()
            break
        elif state in visit:
            # step_record.pop()
            # plan = plan[:-1]
            # visit.remove(state)
            continue
        else:
            # state_record.push(state)
            # plan = step_record.pop()
            # plan += [state[1]]
            visit.append(state)
            # step_record.push(step)

            successors = problem.getSuccessors(state)
            for i in range(len(successors)):
                state_record.push(successors[i][0])
                step_record.push(plan+[successors[i][1]])

    return plan


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    state = problem.getStartState()

    state_record = util.PriorityQueue()
    step_record = util.PriorityQueue()
    # step_record = []
    plan = []
    visit = []


    print "Start:", state

    if problem.isGoalState(state):
        return []
    visit.append(state)

    # state_record.push(state, 0)
    # step_record.push(plan,0)
    successors = problem.getSuccessors(state)
    for i in range(len(successors)):
        state_record.push((successors[i][0], successors[i][2]), successors[i][2])
        step_record.push([successors[i][1]], successors[i][2])

    while state_record.isEmpty() == 0:

        successor = state_record.pop()
        state = successor[0]
        prio_cost = successor[1]
        plan = step_record.pop()
        if problem.isGoalState(state):
            # state_record.push(state)
            # step = step_record.pop()
            # plan += [step]
            # plan = step_record.pop()
            break
        elif state in visit:
            # step_record.pop()
            # plan = plan[:-1]
            # visit.remove(state)
            continue
        else:
            # state_record.push(state)
            # plan = step_record.pop()
            # plan += [state[1]]
            visit.append(state)
            # step_record.push(step)

            successors = problem.getSuccessors(state)
            for i in range(len(successors)):
                if successors[i][0] in visit:
                    continue
                state_record.push((successors[i][0], prio_cost+successors[i][2]), prio_cost+successors[i][2])
                step_record.push(plan+[successors[i][1]], prio_cost+successors[i][2])
                # if state_record.count < 10:
                    # print state_record.heap
    return plan
    # util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    start_state = problem.getStartState()

    state_record = util.PriorityQueue()
    step_record = util.PriorityQueue()
    # step_record = []
    plan = []
    visit = []

    print "Start:", start_state

    if problem.isGoalState(start_state):
        return []
    visit.append(start_state)

    # state_record.push(state, 0)
    # step_record.push(plan,0)
    successors = problem.getSuccessors(start_state)
    for i in range(len(successors)):
        state_record.push((successors[i][0], successors[i][2]),
                          successors[i][2] + heuristic(successors[i][0], problem))
        step_record.push([successors[i][1]],
                         successors[i][2] + heuristic(successors[i][0], problem))

    while state_record.isEmpty() == 0:

        successor = state_record.pop()
        state = successor[0]
        prio_cost = successor[1]
        plan = step_record.pop()
        if problem.isGoalState(state):
            # state_record.push(state)
            # step = step_record.pop()
            # plan += [step]
            # plan = step_record.pop()
            break
        elif state in visit:
            # step_record.pop()
            # plan = plan[:-1]
            # visit.remove(state)
            continue
        else:
            # state_record.push(state)
            # plan = step_record.pop()
            # plan += [state[1]]
            visit.append(state)
            # step_record.push(step)

            successors = problem.getSuccessors(state)
            for i in range(len(successors)):
                if successors[i][0] in visit:
                    continue
                # print successors[i][0], start_state
                state_record.push((successors[i][0], prio_cost + successors[i][2]),
                                  prio_cost + successors[i][2] + heuristic(successors[i][0], problem))
                step_record.push(plan + [successors[i][1]],
                                 prio_cost + successors[i][2] + heuristic(successors[i][0], problem))
                # if state_record.count < 10:
                # print state_record.heap
    return plan


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
