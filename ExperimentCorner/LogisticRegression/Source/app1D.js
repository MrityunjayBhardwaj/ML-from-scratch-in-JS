
// initializing data
const mIrisX = tf.tensor(iris).slice([0,3],[100,1])

// one hot encoded
const mIrisY = tf.tensor(Array(100).fill([1,0],0,50).fill([0,1],50));

const mIrisXWithBias =  mIrisX.concat( tf.ones(mIrisX.shape), axis=1 );


const model = new LogisticRegression();
model.train({x: mIrisXWithBias, y: mIrisY}, params={
            epoch: 2000,
            learningRate: 0.01,
            verbose: 1
});

const mIrisPredY = model.classify(mIrisXWithBias);

const psudoPts = tf.linspace(tf.min(mIrisX).flatten().arraySync()[0], 
                 tf.max(mIrisX).flatten().arraySync()[0],
                 100).expandDims(1);

const psudoPtsWithBias = psudoPts.concat( tf.ones(psudoPts.shape), axis=1);

const psudoPtsLogit = model.logisticFn(0.5, 0 )(psudoPtsWithBias, model.getWeights().mul(1) );



const decisionRegionData = [{
    x : mIrisX.slice([0,0],[50,-1]).flatten().arraySync(),
    y : mIrisPredY.slice([0,0],[50,-1]).matMul(tf.tensor([[1],[0]])).flatten().arraySync(),
    // y : tf.zeros([mIrisX.shape[0]]).flatten().arraySync(),
    type: 'scatter',
    mode: 'markers',
    
    line : {
    // width: .5,
        // smoothing: 0
    },

},
{
    x : mIrisX.slice([50,0],[-1,-1]).flatten().arraySync(),
    y : mIrisPredY.slice([50,0],[-1,-1]).matMul(tf.tensor([[1],[0]])).flatten().arraySync(),
    // y : tf.zeros([mIrisX.shape[0]]).flatten().arraySync(),
    type: 'scatter',
    mode: 'markers',
    
    line : {
    // width: .5,
        // smoothing: 0
    },

},
{
    x :  psudoPts.flatten().arraySync(),
    y : psudoPtsLogit.flatten().arraySync(),
    type: 'scatter'
}
]

Plotly.newPlot('logisticRegressionViz',decisionRegionData,{title: 'input Space'})



// // EXP
// // Visualizing Simoid prob.