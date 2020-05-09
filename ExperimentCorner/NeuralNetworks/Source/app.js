// simulated data
// const sampleSize = 500;
// const simData = {   x : tf.randomNormal([sampleSize/2, 1]).mul(0.5).sub(2.5).concat(tf.randomNormal([sampleSize/2, 1]).mul(0.5).add(2.5)), 
//                     y: tf.ones([sampleSize/2, 1]).concat(tf.zeros([sampleSize/2, 1])) }

let mIrisX = tf.tensor(iris).slice([0,0],[150,-1]);
// one hot encoded
let mIrisY = tf.tensor( Array(150).fill([1,0, 0],0,50).fill([0,1, 0],50, 100).fill([0, 0, 1], 100, 150) );

// let mIrisY = tf.tensor( Array(150).fill([1,0,0], 0, 50).fill([0,1,0], 50, 100).fill([0,0,1], 100) );
const standardDataX = normalizeData(mIrisX, 1)

let [trainData, testData] = trainTestSplit(standardDataX, mIrisY, 0.9);

const model = new NeuralNetworks([10,10,10],
     {epoch: 100, learningRate: 0.000002, batchSize: 1.0 /* percent of data */ ,verbose: true},
     );

// model.train(trainData);
// const predY = model.test(trainData);

// TODO: MAKE THE CODE MORE PROFESSIONAL AND APPLICABLE.



let classesData = {
                   zero:  [],
                   one:   [],
                   two:   [],
                   three: [],
                   four:  [],
                   five:  [],
                   six:   [],
                   seven: [],
                   eight: [],
                   nine:  [],
               };
// let trainY = [];

const classNames = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];

let dataX, dataY,myTrainData, myTestData, predY;

d3.csv("../../dependency/data/mnist/mnist_test.csv", function(d) {
     //   console.log(d)

     const currImg = [];

     // fetching all the rows
     for(let u in d){
          if (u != 'label')
               currImg.push(d[u]*1);

     }

     // console.log(classesData[classNames[d['label']*1]])

     classesData[classNames[d['label']*1]].push(currImg)

   }).then(
        () =>{

// // convert to tensor
          for(let digit in classesData){

               console.log(tf.tensor(classesData[digit]).shape)
               classesData[digit] = tf.tensor(classesData[digit])
          }
               dataX = classesData.two.concat(classesData.five, axis=0);
               dataY = tf.tile(tf.tensor([1, 0]).expandDims(1).transpose(), [classesData.two.shape[0],1] ).concat(
                       tf.tile(tf.tensor([0, 1]).expandDims(1).transpose(), [classesData.five.shape[0],1] )
                         );
           [myTrainData, myTestData]  = trainTestSplit( dataX ,  dataY, .05);

          model.train(myTrainData);
          predY = model.test(myTrainData);

          console.log("accuracy:-")
          tf.abs(tf.argMax(predY, 1).equal(tf.argMax(myTrainData.y,1))).sum().div(myTrainData.y.shape[0]).print()

          }
     )



// creating data

// currently only discriminate between 2 digits :- '2' and '5'




