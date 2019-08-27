let mIrisX = tf.tensor(iris).slice([0,1],[150,-1])
// one hot encoded
// let mIrisY = tf.tensor( Array(100).fill([1,0],0,50).fill([0,1],50) );

let mIrisY = tf.tensor( Array(150).fill([1,0,0], 0, 50).fill([0,1,0], 50, 100).fill([0,0,1], 100) );

const standardDataX = normalizeData(mIrisX, 1)

const {0: trainData,1: testData} = trainTestSplit(standardDataX,mIrisY,2/3);

const model = new LDA;

model.train(trainData);

const predicted = model.classify(testData.x);