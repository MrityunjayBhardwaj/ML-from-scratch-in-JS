
// initializing data
const mIrisX = tf.tensor(iris).slice([0,1],[100,2])

// creating one hot encoded y vector.
const mIrisY = tf.tensor(Array(100).fill([1,0],0,50).fill([0,1],50));

// training our model on this dataset.
const model = new perceptron();
model.train({x: mIrisX, y: mIrisY});

// testing
// model.test()
