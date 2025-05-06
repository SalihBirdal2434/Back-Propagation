# Symbolic Back-Propagation Demo (SymPy)

This repo contains a single Python script that

1. **Defines** a tiny fully-connected neural network with two inputs, one hidden layer
   of two neurons, and one output neuron (all linear activations).
2. **Builds** the Mean-Squared-Error loss  
   \\[
   
     L\;=\;\bigl(\hat y - y_{\text{Actual}}\bigr)^2
   \\]
   where \\( \hat y \\) is the model prediction.
4. **Computes** exact gradients \\( \partial L / \partial w_i \\) and  
   \\( \partial L / \partial b_i \\) symbolically using
   [SymPy](https://www.sympy.org).
5. **Evaluates** those gradients for a concrete numeric example  
   \\(x_1=1,\,x_2=0,\,y=2\\).
