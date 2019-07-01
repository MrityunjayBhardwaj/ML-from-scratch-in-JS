// Generate some synthetic data for Regression.
let trainX = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
let trainY = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);

let matrixX = tf.tensor([[1,2],[2,5],[7,8],[12,15]]);

const weights = tf.tensor([3,2,-3]).expandDims(1);

const matrixXMod = matrixX.concat( tf.ones([matrixX.shape[0],1]), axis=1);
matrixX.print();

let matrixY = matrixXMod.matMul( weights );

let modelLR = new LinearRegression();
modelLR.train( { x: matrixX, y: matrixY } );


/* Visualizing */




 /* Classification */

trainX = tf.tensor([[1,9,3,4,6,7,8,2],[6,4,8,9,1,2,1,7]]).transpose();
trainY = tf.tensor([[1,0],[0,1],[1,0],[1,0],[0,1],[0,1],[0,1],[1,0]]);

// trainX.print();
// trainY.print();
// // Generate some synthetic data for Regression.
// const trainX = tf.tensor([])

testX = tf.tensor([[1,7],[6,2]]);
testY = tf.tensor([[1,0],[0,1]]);