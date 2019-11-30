import numpy as np

np.random.seed(1)

def relu(x):
    return (x > 0) * x # returns x if x > 0
                       # return 0 otherwise

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid2deriv(prepro):
    return (sigmoid(prepro)*( 1 - sigmoid(prepro)))

def relu2deriv(output):
    return output>0 # returns 1 for input > 0
                    # return 0 otherwise

streetlights = np.array( [[ 1, 0, 1 ],
                          [ 0, 1, 1 ],
                          [ 0, 0, 1 ],
                          [ 1, 1, 1 ] ] )

walk_vs_stop = np.array([[ 1, 1, 0, 0]]).T
    
alpha = 0.2
hidden_size = 4

# weights_0_1 = 2*np.random.random((3,hidden_size)) - 1
# weights_1_2 = 2*np.random.random((hidden_size,1)) - 1

weights_0_1 = np.array([[-0.16595599,  0.44064899, -0.99977125, -0.39533485],
                        [-0.70648822, -0.81532281, -0.62747958, -0.30887855],
                        [-0.20646505,  0.07763347, -0.16161097,  0.370439  ]]);

weights_1_2 = np.array([[-0.5910955], [0.75623487],[-0.94522481],[0.34093502]])

for iteration in range(1000):
   layer_2_error = 0
   for i in range(len(streetlights)):
      layer_0 = streetlights[i:i+1]
      layer_1 = sigmoid(np.dot(layer_0,weights_0_1))
      layer_2 = np.dot(layer_1,weights_1_2)

      layer_2_error += np.sum((layer_2 - walk_vs_stop[i:i+1]) ** 2)

      layer_2_delta = (layer_2 - walk_vs_stop[i:i+1])
      layer_1_delta = layer_2_delta.dot(weights_1_2.T)*sigmoid2deriv(np.dot(layer_0,weights_0_1))

      weights_1_2 -= alpha * layer_1.T.dot(layer_2_delta)
      weights_0_1 -= alpha * layer_0.T.dot(layer_1_delta)

   if(iteration % 10 == 9):
      print("Error:" + str(layer_2_error))

