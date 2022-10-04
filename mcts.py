import random
import math
from anytree import Node, RenderTree
import numpy as np
from pprint import pprint
import itertools
class MCTS:
    def __init__(self, terminal_condition, simulate, expand, C=2, root_name=""):
        self.terminal_condition = terminal_condition
        self.simulate = simulate
        self.expand = expand
        self.C = C
        self.root = Node(root_name, t=0, n=0, ucb1=math.inf)
    
    def __is_leaf(node):
        return not node.children

    def tree_policy(self, node):
        if MCTS.__is_leaf(node):
            if node.n == 0 or self.terminal_condition(node):
                return node
            else:
                return self.expand(node)
        else:
            max_ = node.children[0]
            for i in range(1, len(node.children)):
                if node.children[i].ucb1 > max_.ucb1:
                    max_ = node.children[i]
            return self.tree_policy(max_) 

    def backprop(self, node, value):
        current = node
        while current != None:
            current.t += value
            current.n += 1
            # select the parent depending if the node is root or not
            parent = current.parent if current.parent else current
            # ucb1 calculation
            current.ucb1 = ( (current.t / current.n) )
            # exploration term
            current.ucb1 += self.C * math.sqrt(2 * math.log(parent.n)/current.n)            
            current = current.parent

    def run(self, max_iter=1000):
        for i in itertools.count():
            if max_iter :
                if i > max_iter: break
            v_l = self.tree_policy(self.root)
            value = self.simulate(v_l, i)
            self.backprop(v_l, value)


    def bestchild(self):
        current = self.root
        while not MCTS.__is_leaf(current):
            max_ = current.children[0]
            for i in range(1, len(current.children)):
                if current.children[i].ucb1 > max_.ucb1:
                    max_ = current.children[i]
            current = max_
        return current


    def bestchild_from(self, node):
        current = node
        while not MCTS.__is_leaf(current):
            max_ = current.children[0]
            for i in range(1, len(current.children)):
                if current.children[i].ucb1 > max_.ucb1:
                    max_ = current.children[i]
            current = max_
        return current


    def __str__(self):
        return str(RenderTree(self.root))



