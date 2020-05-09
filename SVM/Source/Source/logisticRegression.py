import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# dataset = pd.read_csv("ex2data1.csv")

# X = dataset.iloc[:,0:2].values
# Y = dataset.iloc[:,2:3].values




def sigmoid(a):
    b = 1 + pow(np.e, -a)
    b = 1/b
    return b

def a(x, weights, bias):
    p = np.matmul(weights, x) + bias
    return p

def loss_fn(predY, y):
    j = -( y*np.log( predY ) + (1-y)*np.log( 1-predY ) )
    return j

def weightDerivative(predY, x, y):
    return (y - predY)*x


originalWeights = np.expand_dims(np.array([3, 5]), 1)
originalBias = 2

# Creating synthetica data:=
dataSize = 100
dataX = ( np.random.normal(0, 1, [dataSize,2]) )

dataY = (sigmoid(np.matmul(dataX, originalWeights) + originalBias) > 0.5)*1

X = dataX
Y = dataY

# user-defined constants
alpha = 0.009
normConst=1/len(X)
dataLen = len(X)
epoch = 1000

#  initialization
theta_0 = 2
theta_1 = 3
theta_2 = 4


weights = np.array([theta_1,theta_2]).reshape([1, 2]) 
bias = np.array( theta_0).reshape([1, 1])

cost = 0.0

# training loop:-

#  testing for one value

# cIndex = 1
# cX = np.expand_dims(X[cIndex], 1)
# cY = np.expand_dims(Y[cIndex], 1)

# # predicting value
# hyperplane = a(cX, weights, bias)
# predY = sigmoid(hyperplane)

# cLoss = loss_fn( predY, cY )

# # calculating derivative:-
# xOne = np.tile( np.array([1]), [len(X)*0 + 1, 1])
# modX = np.concatenate((xOne, cX))

# cDx = weightDerivative(predY, modX, cY)

# # updading our thetas

# weights = weights - alpha * cDx[1:]
# bias = bias - alpha * cDx[0]



#  for all the shit:-


for i in range(epoch):

    cummDx = 0 # cummulative derivative
    costVal = 0
    for j in range(dataLen):
        cIndex = j
        cX = np.expand_dims(X[cIndex], 1)
        cY = np.expand_dims(Y[cIndex], 1)

        # predicting value
        hyperplane = a(cX, weights, bias)
        predY = sigmoid(hyperplane)

        cLoss = loss_fn( predY, cY )
        costVal += cLoss

        # calculating derivative:-
        xOne = np.tile( np.array([1]), [len(X)*0 + 1, 1])
        modX = np.concatenate((xOne, cX))

        cDx = weightDerivative(predY, modX, cY)
        cummDx += cDx

    # updading our thetas


    hyperplane = a(X.T, weights, bias)
    predY = (sigmoid(hyperplane) > 0.5)*1

    print(i, np.sum(Y)/dataLen, np.sum(np.abs(Y - predY.T))/dataLen, costVal/dataLen, weights, bias)


    weights = weights + (alpha * (cummDx[1:]/dataLen)).T
    bias = bias + (alpha * cummDx[0]).T

    if ((costVal/dataLen) < 0.1):
        break

print('awesome!!', weights, bias)
# for i in range(epoch):
#     for i in range(len(X)):



decision_boundary_p1_x = np.min([x[:,0],y[:,0]])
decision_boundary_p2_x = np.max([x[:,0],y[:,0]])
# Line form: (-a[0] * x - b ) / a[1]
decision_boundary_p1_y = (-a.value[0]*decision_boundary_p1_x + b.value ) / a.value[1]
decision_boundary_p2_y = (-a.value[0]*decision_boundary_p2_x + b.value ) / a.value[1]

margin_left_p1_y = (-a.value[0]*decision_boundary_p1_x + b.value + margin_length ) / a.value[1]
margin_left_p2_y = (-a.value[0]*decision_boundary_p2_x + b.value + margin_length ) / a.value[1]
margin_right_p1_y = (-a.value[0]*decision_boundary_p1_x + b.value - margin_length ) / a.value[1]
margin_right_p2_y = (-a.value[0]*decision_boundary_p2_x + b.value - margin_length ) / a.value[1]

data = [
        go.Scatter(
          x = x[:,0].flatten(),
          y = x[:,1].flatten(),
          mode="markers",
          name="class 1 data",   
          marker=dict(color="blue")  
                 
        ),
        go.Scatter(
          x = y[:,0].flatten(),
          y = y[:,1].flatten(),
           mode="markers",
          name="class 2 data",  
          marker=dict(color="orange")               
        ),
        go.Scatter(
            x= [decision_boundary_p1_x.flatten()[0], decision_boundary_p2_x.flatten()[0]],
            y= [decision_boundary_p1_y, decision_boundary_p2_y],
            line=dict(color="black"),
            name="decision Boundary"
        ),
        go.Scatter(
            x = [decision_boundary_p1_x, decision_boundary_p2_x],
            y = [margin_left_p1_y, margin_left_p2_y],
            legendgroup= 'a',
            name="margins",
            line=dict(color="gray", dash="dash"),
            showlegend = False,
            
        ),
        go.Scatter(
            x = [decision_boundary_p1_x, decision_boundary_p2_x],
            y = [margin_right_p1_y, margin_right_p2_y],
            legendgroup= 'a',
            name="margins",
            line=dict(color="gray", dash='dash'),
            showlegend = False,
            
        )
]

# plt.plot(data=data)
fig = go.Figure(data)


# some visual stuffs.
fig.update_layout(title='Support Vector Machine', autosize=False,
                  # scene_camera_eye=dict(x=optimalPt[0]*1, y=optimalPt[1]*1, z=optimalVal*1),
                  width=600, height=600,
                  margin=dict(l=65, r=50, b=65, t=90),
                  scene={
                      "xaxis": {"title": "x"},
                      "yaxis": {"title": "y"},
                  }
                  
)

fig.update_yaxes(range=(-4, 6))


# [decision_boundary_p1_x,decision_boundary_p2_x],[margin_left_p1_y[0,0],margin_left_p2_y[0,0]]
fig.show()
