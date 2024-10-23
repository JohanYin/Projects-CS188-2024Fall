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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** MY CODE HERE ***"
        distToNearestFood = min([util.manhattanDistance(newPos, foodPos) for foodPos in currentGameState.getFood().asList()])
        foodScore = 1 / (1 + distToNearestFood)

        distToNearestGhost = min([util.manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates])
        if min(newScaredTimes) > distToNearestGhost:
            ghostScore = float('inf')
        elif distToNearestGhost <= 1:
            ghostScore = float('-inf')
        else:
            ghostScore = 0

        capsuleList = [util.manhattanDistance(newPos, capsulePos) for capsulePos in currentGameState.getCapsules()]
        if not capsuleList:
            capsuleScore = 0
        else:
            distToNearestCapsule = min(capsuleList)
            capsuleScore = 2 / (1 + distToNearestCapsule)

        return successorGameState.getScore() + foodScore + ghostScore + capsuleScore

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        "*** MY CODE HERE ***"
        def maxValue(state, agent, depth):
            legalActions = state.getLegalActions(agent)
            v = float('-inf')
            for action in legalActions:
                successor = state.generateSuccessor(agent, action)
                v = max(v, value(successor, 1, depth))
            return v

        def minValue(state, agent, depth):
            legalActions = state.getLegalActions(agent)
            v = float('inf')
            for action in legalActions:
                successor = state.generateSuccessor(agent, action)
                if agent + 1 == state.getNumAgents():
                    v = min(v, value(successor, 0, depth + 1))
                else:
                    v = min(v, value(successor, agent + 1, depth))
            return v

        def value(state, agent, depth):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if agent == 0:
                return maxValue(state, agent, depth)
            else:
                return minValue(state, agent, depth)

        legalActions = gameState.getLegalActions(0)
        successors = [gameState.generateSuccessor(0, action) for action in legalActions]
        maxVal = float('-inf')
        maxIdx = 0
        for i, successor in enumerate(successors):
            actionVal = value(successors[i], 1, 0)
            if actionVal > maxVal:
                maxVal, maxIdx = actionVal, i

        return legalActions[maxIdx]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** MY CODE HERE ***"
        def maxValue(state, agent, depth, alpha, beta):
            legalActions = state.getLegalActions(agent)
            v = float('-inf')
            for action in legalActions:
                successor = state.generateSuccessor(agent, action)
                v = max(v, value(successor, 1, depth, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def minValue(state, agent, depth, alpha, beta):
            legalActions = state.getLegalActions(agent)
            v = float('inf')
            for action in legalActions:
                successor = state.generateSuccessor(agent, action)
                if agent + 1 == state.getNumAgents():
                    v = min(v, value(successor, 0, depth + 1, alpha, beta))
                else:
                    v = min(v, value(successor, agent + 1, depth, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        def value(state, agent, depth, alpha, beta):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if agent == 0:
                return maxValue(state, agent, depth, alpha, beta)
            if agent > 0:
                return minValue(state, agent, depth, alpha, beta)

        alpha = float('-inf')
        beta = float('inf')
        legalActions = gameState.getLegalActions(0)
        successors = [gameState.generateSuccessor(0, action) for action in legalActions]
        maxVal = float('-inf')
        maxIdx = 0
        for i, successor in enumerate(successors):
            actionVal = value(successor, 1, 0, alpha, beta)
            if actionVal > maxVal:
                maxVal, maxIdx = actionVal, i
                alpha = actionVal

        return legalActions[maxIdx]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** MY CODE HERE ***"
        def maxValue(state, agent, depth):
            legalActions = state.getLegalActions(agent)
            v = float('-inf')
            for action in legalActions:
                successor = state.generateSuccessor(agent, action)
                v = max(v, value(successor, 1, depth))
            return v

        def expValue(state, agent, depth):
            legalActions = state.getLegalActions(agent)
            successors = [state.generateSuccessor(agent, action) for action in legalActions]
            v = float(0)
            for successor in successors:
                if agent + 1 == state.getNumAgents():
                    v += value(successor, 0, depth + 1)
                else:
                    v += value(successor, agent + 1, depth)
            return v / len(successors)

        def value(state, agent, depth):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if agent == 0:
                return maxValue(state, agent, depth)
            if agent > 0:
                return expValue(state, agent, depth)

        legalActions = gameState.getLegalActions(0)
        successors = [gameState.generateSuccessor(0, action) for action in legalActions]
        maxVal = float('-inf')
        maxIdx = 0
        for i, successor in enumerate(successors):
            actionVal = value(successors[i], 1, 0)
            if actionVal > maxVal:
                maxVal, maxIdx = actionVal, i

        return legalActions[maxIdx]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    
    "*** MY CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    x, y = pos
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    foodList = [util.manhattanDistance(pos, foodPos) for foodPos in currentGameState.getFood().asList()]

    # food-gobbling
    if foodList:
        distToNearestFood = min(foodList)
    else:
        distToNearestFood = 0
    foodScore = 1 / (1 + distToNearestFood)

    # ghost-hunting (ghost-dodging)
    distToNearestGhost = min(
        [util.manhattanDistance(pos, ghostState.getPosition()) for ghostState in ghostStates])
    if min(scaredTimes) > distToNearestGhost:
        ghostScore = float('inf')
    elif distToNearestGhost <= 1:
        ghostScore = float('-inf')
    else:
        ghostScore = 0

    # pellet-nabbing
    capsuleList = [util.manhattanDistance(pos, capsulePos) for capsulePos in currentGameState.getCapsules()]
    if not capsuleList:
        capsuleScore = 0
    else:
        distToNearestCapsule = min(capsuleList)
        capsuleScore = 2 / (1 + distToNearestCapsule)

    return currentGameState.getScore() + foodScore + ghostScore + capsuleScore

# Abbreviation
better = betterEvaluationFunction
