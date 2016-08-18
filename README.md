# Game-Learning
Project to learn and play games using Deep Learning. (a week old)

# Sandwich 
Sandwich is a lib to build a Neural Network. Much like 'lasange'.

# Current status
- Testing by training a Net to recognize handwritten digits.
- Built basic layers: InputLayer, FullConn, Conv2D, Pool, Flatten

# Results
- A single layer and a two layer Fully connected network give an accuracy of about 90%.
- A Conv Neural Network however, seems to be stuck at a local minima. (tuning the parameters doesn't seem to work ...)

# Finding your way around
+ *digits.tra*: Training dataset for handwritten digits.
+ *digits.va* : Validation/Evaluation dataset for handwritten digits.
+ *sandwich*  : Neural network library built using theano, numpy.
+ *digits.py* : Script to test *sandwich* by training nets for handwritten digits.

