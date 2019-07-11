
const x = tf.tensor([[1],[2]]);

// generate meshgrid
const grid = meshGridRange(range={x: {min: -1.1, max: +1.1},y: {min: -1.1, max: +1.1}},division=30);

// calculate pNorm for each value of meshgrid

