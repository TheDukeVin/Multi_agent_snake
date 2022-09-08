# Multi-agent Snake
This project aims to apply tree-based search algorithms to the game of multi-agent snake to find approximate Nash Equilibria. Our current implementation uses a variation of the AlphaZero algorithm to achieve state-of-the-art performance in single-agent Snake.

To run, use the following command to compile all files:
'''
g++ -std=c++11 -pthread main.cpp data.cpp convNet.cpp environment.cpp InputLayerCode.cpp trainer.cpp
'''
and execute './a.out'.
