import util

"""
Data sturctures we will use are stack, queue and priority queue.

Stack: first in last out
Queue: first in first out
    collection.push(element): insert element
    element = collection.pop() get and remove element from collection

Priority queue:
    pq.update('eat', 2)
    pq.update('study', 1)
    pq.update('sleep', 3)
pq.pop() will return 'study' because it has highest priority 1.

"""

"""
problem is a object has 3 methods related to search state:

problem.getStartState()
Returns the start state for the search problem.

problem.isGoalState(state)
Returns True if and only if the state is a valid goal state.

problem.getChildren(state)
For a given state, this should return a list of tuples, (next_state,
step_cost), where 'next_state' is a child to the current state,
and 'step_cost' is the incremental cost of expanding to that child.

"""


def myDepthFirstSearch(problem):
    visited = {}
    frontier = util.Stack()
    frontier.push((problem.getStartState(), None))

    while not frontier.isEmpty():
        state, prev_state = frontier.pop()
        if problem.isGoalState(state):
            solution = [state]
            while prev_state != None:
                solution.append(prev_state)
                prev_state = visited[prev_state]
            return solution[::-1]
        if state not in visited:
            visited[state] = prev_state
            for next_state, step_cost in problem.getChildren(state):
                frontier.push((next_state, state))
    return []


def myBreadthFirstSearch(problem):
    # YOUR CODE HERE
    visited = {}
    queue = util.Queue()
    queue.push((problem.getStartState(), None))

    while not queue.isEmpty():
        state, prev_state = queue.pop()
        if problem.isGoalState(state):
            solution = [state]
            while prev_state != None:
                solution.append(prev_state)
                prev_state = visited[prev_state]
            return solution[::-1]
        if state not in visited:
            visited[state] = prev_state
            for next_state, step_cost in problem.getChildren(state):
                queue.push((next_state, state))
    return []


def myAStarSearch(problem, heuristic):
    # YOUR CODE HERE
    visited = {}
    cost = {}
    pqueue = util.PriorityQueue()
    cost[problem.getStartState()] = 0.0 # start
    pqueue.push((problem.getStartState(), None), heuristic(problem.getStartState()))

    while not pqueue.isEmpty():
        state, prev_state = pqueue.pop()
        if problem.isGoalState(state):
            solution = [state]
            while prev_state != None:
                solution.append(prev_state)
                prev_state = visited[prev_state]
            return solution[::-1]
        if state not in visited:
            visited[state] = prev_state
            for next_state, step_cost in problem.getChildren(state):
                cost[next_state] = cost[state] + step_cost
                pqueue.push((next_state, state), heuristic(
                    next_state)+cost[next_state])
    return []


"""
Game state has 4 methods we can use.

state.isTerminated()
Return True if the state is terminated. We should not continue to search if the state is terminated.

state.isMe()
Return True if it's time for the desired agent to take action. We should check this function to determine whether an agent should maximum or minimum the score.

state.getChildren()
Returns a list of legal state after an agent takes an action.

state.evaluateScore()
Return the score of the state. We should maximum the score for the desired agent.

"""


class MyMinimaxAgent():
    def __init__(self, depth):
        self.depth = depth

    def minimax(self, state, depth):
        if state.isTerminated():
            return None, state.evaluateScore()

        best_state = None
        if state.isMe():
            best_score = -float('inf')
        else:
            best_score = float('inf')
        if state.isMe():
            # print('Me depth:', depth)
            if depth == 0:
                return state, state.evaluateScore()

        for child in state.getChildren():
            # YOUR CODE HERE
            if state.isMe():
                child_state, child_score = self.minimax(child, depth-1)
                if child_score > best_score:
                    best_score = child_score
                    best_state = child
            else:
                child_state, child_score = self.minimax(child, depth)
                if child_score < best_score:
                    best_score = child_score
                    best_state = child
        return best_state, best_score

    def getNextState(self, state):
        best_state, _ = self.minimax(state, self.depth)
        return best_state


class MyAlphaBetaAgent():
    def __init__(self, depth):
        self.depth = depth

    def alphaBeta(self, state, depth, alpha, beta):
        # YOUR CODE HERE
        if state.isTerminated():
            return None, state.evaluateScore()

        best_state = None
        if state.isMe():
            best_score = -float('inf')
        else:
            best_score = float('inf')
        if state.isMe():
            # print('Me depth:', depth)
            if depth == 0:
                return state, state.evaluateScore()

        for child in state.getChildren():
            if state.isMe():
                child_state, child_score = self.alphaBeta(
                    child, depth, alpha, beta)
                if child_score > best_score:
                    best_score = child_score
                    best_state = child
                if best_score > beta:
                    return best_state, best_score
                alpha = max(alpha, best_score)
            else:
                if child.isMe():
                    child_state, child_score = self.alphaBeta(
                        child, depth-1, alpha, beta)
                else:
                    child_state, child_score = self.alphaBeta(
                        child, depth, alpha, beta)
                if child_score < best_score:
                    best_score = child_score
                    best_state = child
                if best_score < alpha:
                    return best_state, best_score
                beta = min(beta, best_score)
        return best_state, best_score

    def getNextState(self, state):
        # YOUR CODE HERE
        best_state, _ = self.alphaBeta(
            state, self.depth, -float('inf'), float('inf'))
        return best_state
