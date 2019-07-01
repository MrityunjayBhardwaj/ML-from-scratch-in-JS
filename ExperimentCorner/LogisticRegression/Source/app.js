
// initializing data
const mIrisX = tf.tensor(iris).slice([0,0],[100,2])
// one hot encoded
const mIrisY = Array(100).fill([1,0],0,50).fill([0,1],50);