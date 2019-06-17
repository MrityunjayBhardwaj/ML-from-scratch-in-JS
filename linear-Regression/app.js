// Generate some synthetic data for Regression.
let trainX = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
let trainY = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);

// let modelLR = new LinearRegression(trainX,trainY);
// modelLR.train();

// let modelPR = new PolynomialRegression(trainX,trainY);
// modelPR.train();

// /* Classification */

trainX = tf.tensor([[1,9,3,4,6,7,8,2],[6,4,8,9,1,2,1,7]]).transpose();
trainY = tf.tensor([[1,0],[0,1],[1,0],[1,0],[0,1],[0,1],[0,1],[1,0]]);

trainX.print();
trainY.print();
// // Generate some synthetic data for Regression.
// const trainX = tf.tensor([])

testX = tf.tensor([[1,7],[6,2]]);
testY = tf.tensor([[1,0],[0,1]]);

let modelBC = new BayesClassifier(trainX,trainY);
modelBC.train();
modelBC.test(testX);