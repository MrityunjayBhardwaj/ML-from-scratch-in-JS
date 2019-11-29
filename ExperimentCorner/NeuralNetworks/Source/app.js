let mIrisX = tf.tensor(iris).slice([0,0],[100,1]);


// simulated data

const sampleSize = 500;

const simData = {   x : tf.randomNormal([sampleSize/2, 1]).mul(0.5).sub(2.5).concat(tf.randomNormal([sampleSize/2, 1]).mul(0.5).add(2.5)), 
                    y: tf.ones([sampleSize/2, 1]).concat(tf.zeros([sampleSize/2, 1])) }

// one hot encoded
let mIrisY = tf.tensor( Array(100).fill([1,0],0,50).fill([0,1],50) );

// let mIrisY = tf.tensor( Array(150).fill([1,0,0], 0, 50).fill([0,1,0], 50, 100).fill([0,0,1], 100) );
const standardDataX = normalizeData(mIrisX, 1)

const trainData = { x : standardDataX.slice([0,0],[Math.floor(standardDataX.shape[0]*2/3), -1]),
                    y: mIrisY.slice([0,0],[Math.floor(standardDataX.shape[0]*2/3), 1]) };

const testData  = { x: standardDataX.slice([Math.floor(standardDataX.shape[0]*2/3), 0],[-1, -1]),
                    y: mIrisY.slice([Math.floor(standardDataX.shape[0]*2/3), 0],[-1, 1]) };

// const {0: trainData,1: testData} = trainTestSplit(standardDataX, mIrisY.slice([0,0],[-1,1]), 2/3);

// const classwiseDataSplit

const streetlights = tf.tensor( [[ 1, 0, 1 ],
                          [ 0, 1, 1 ],
                          [ 0, 0, 1 ],
                          [ 1, 1, 1 ] ] )

const walk_vs_stop = tf.tensor([[ 1, 1, 0, 0]]).transpose()

const gdltrain = {x:streetlights, y: walk_vs_stop}

const model = new NeuralNetworks([3,3,3],
     {epoch: 100, learningRate: 0.001, batchSize: 1.0 /* percent of data */ },
     );

model.train(trainData);
model.test(testData);

// TODO: MAKE THE CODE MORE PROFESSIONAL AND APPLICABLE.


// testing:
// model.test(testData)


// const originalWeights = 5;
// let trainX = tf.randomUniform();
// let trainY = trainX.mul(originalWeights);


// const regModel = new NeuralNetworks([3,3]);

// model.train(trainX);






















// // const predicted = model.classify(testData.x);

// let predY = tf.tensor([]);
// for(let i=0;i< testData.x.shape[0]; i++){

//   const currPtX = testData.x.slice([i, 0],[1, -1]);
//   predY = predY.concat( model.classify( currPtX ) );

// }

// /* Visualizing Decision Boundary in the input space */


// const grid = meshGridRange(
//   (range = { 
    
//         x: { min: tf.min(trainData.x, axis=0).arraySync()[1],
//              max: tf.max(trainData.x, axis=0).arraySync()[1] },
    
//         y: { min: tf.min(trainData.x, axis=0).arraySync()[0],
//              max: tf.max(trainData.x, axis=0).arraySync()[0] }

//     }),
//   (division = 10)
// );

// const pNormGrid = grid.map(a => {
//   const f = tf
//     .tensor(a)
//     .transpose()
//     .arraySync();
//   const w = f.map(b => {
//     // return 1*(pNorm(tf.tensor(b).expandDims(1), p=2).flatten().arraySync()[0] )

//     return (
//       1 *
//       model
//         .classify(
//           tf.tensor(b).expandDims(1).transpose(),
//         )
//         .flatten()
//         .arraySync()[0]
//     );

//     // return 1*(inducedMatrixNorm(tf.tensor([[1, 2],[0, 2]]),tf.tensor(b).expandDims(1),p=2).flatten().arraySync()[0] )
//   });
//   return w;
// });



// const pNormVizData = [
//   {
//     x: grid[0][0],
//     y: grid[0][0],
//     z: pNormGrid,
//     type: 'contour',

//     colorscale: [
//       [0, darkModeCols.blue()],
//       [0.25, darkModeCols.purple()],
//       [0.5, darkModeCols.magenta()],
//       [0.75, darkModeCols.yellow()],
//       [1, darkModeCols.red()]
//     ],

//     contours: {
//         start: 0,
//         end: 2,
//         size: 1 
//      },
//   //   line: {}
//   },

//   {
//       x: standardDataX.slice([0,0],[50,-1]).transpose().arraySync()[0],
//       y: standardDataX.slice([0,0],[50,-1]).transpose().arraySync()[1],
//       type: 'scatter',
//       mode: 'markers'


//   },
//   {
//       x: standardDataX.slice([50,0],[100,-1]).transpose().arraySync()[0],
//       y: standardDataX.slice([50,0],[100,-1]).transpose().arraySync()[1],
//       type: 'scatter',
//       mode: 'markers'


//   },
//   {
//       x: standardDataX.slice([100,0],[-1,-1]).transpose().arraySync()[0],
//       y: standardDataX.slice([100,0],[-1,-1]).transpose().arraySync()[1],
//       type: 'scatter',
//       mode: 'markers'


//   }
// ];

// const layoutSetting = {
//   title: "Quadratic Discriminant Analysis",
//   font: {
//     size: 15,
//     color: "white",
//     family: "Helvetica"
//   },
//   paper_bgcolor: "#222633"
// };

// Plotly.newPlot("QDAViz", pNormVizData, layoutSetting);
