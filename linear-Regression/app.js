// Generate some synthetic data for training.
const trainX = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
const trainY = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);


let model = new PolynomialRegression(trainX,trainY);
model.train();

