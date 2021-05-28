---
documentclass: ctexart # for linux
title: 人工智能实验一
author: PB18000221 袁一玮
# date: 5 月 11 日
# CJKmainfont: "Microsoft YaHei" # for win
# CJKmainfont: "KaiTi" # for win
---

## 实验环境

Python 3.7.3、Kernel 4.19.181-1、Debian 10 (buster)

## 实验内容

### BFS

类似 DFS，使用 `Queue` 来存储访问到的路径，`visited` 字典保存访问到的节点。

每次从队列中弹出节点，若是目标节点，则通过 visited 获取到到达目标节点的路径返回；如果该节点还未访问过，则将该节点加入 visited 中。循环上述过程直至队列为空。

```python
def myBreadthFirstSearch(problem):
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
```

### A\*

A\* 算法使用优先队列来排序，按照 f(n)=h(n)+g(n) 排序。与 BFS 类似，每次从优先队列弹出第一个元素，在 `visited` 字典中保存访问的节点，`cost` 字典保存了到每个节点的距离。

初始时 `cost` 字典中只有初始节点，距离是 0；在新节点上使用 `cost[next_state] = cost[state] + step_cost` 记录距离。每次从优先队列弹出节点，若是目标节点则通过 `visited` 找到路径并返回。循环上述过程直至优先队列为空。

```python
def myAStarSearch(problem, heuristic):
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
```

### Minimax

```python
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
```

### Alpha-beta 剪枝

alpha-beta 剪枝与 Minimax 类似。在遍历决策树时，如果发现我方 Agent 找到的子节点的最大值大于 beta 时，遍历子节点无意义；同理，如果发现对方 Agent 找到的子节点最小值小于 alpha，也可以进行剪枝。

书上算法有坑，判断当前最大(小)值和 beta(alpha) 的关系时不要用 >=(<=) 而是要用 >(<)，否则可能会错误剪枝。

```python
def alphaBeta(self, state, depth, alpha, beta):
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
```

## 实验结果

每次测试时运行 `bash test.sh`，test.sh 中测试了三个 search 和两个 agent 的策略

```shell
python3 search/autograder.py -q q1
python3 search/pacman.py -l mediumMaze -p SearchAgent --frameTime 0
python3 search/autograder.py -q q2
python3 search/pacman.py -l mediumMaze -p SearchAgent -a fn=bfs --frameTime 0
python3 search/autograder.py -q q3
python3 search/pacman.py -l mediumMaze -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic --frameTime 0
python3 multiagent/autograder.py -q q2
python3 multiagent/autograder.py -q q3
python3 multiagent/pacman.py -p AlphaBetaAgent -l mediumClassic --frameTime 0
```

所有 case 通过，尽管 agent 在最后一个测试时会失败

![ai-test1](assets/ai1.png)

![ai-test2](assets/ai2.png)
