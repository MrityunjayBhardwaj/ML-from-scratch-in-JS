


const X = tf.randomNormal([5,5]);
const a = tf.tensor([1,2,3,4,5]).expandDims(1);

const output = insert2Tensor(X,a,[0, 2]);

output.print();
