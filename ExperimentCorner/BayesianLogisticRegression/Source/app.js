// initializing data
const IrisX = tf.tensor(iris).slice([0, 1], [100, 2]);

// creating one hot encoded y vector.
const IrisY = tf.tensor(
  Array(100)
    .fill([1, 0], 0, 50)
    .fill([0, 1], 50)
);

// normalizing data
const mIrisX = normalizeData(IrisX);

// generate meshgrid
const grid = meshGridRange(
  (range = { x: { min: -2.1, max: +2.1 }, y: { min: -2.1, max: +2.1 } }),
  (division = 50)
);

const tts = trainTestSplit(mIrisX, IrisY, .9);

const trainX = tts[0].x;
const trainY = tts[0].y;

const testX = tts[1].x;
const testY = tts[1].y;

const model = new bayesianLogisticRegression();

// training BLR:-

model.train({x: trainX, y: trainY});

model.test( { x: testX.slice([0,0],[1, -1]) });