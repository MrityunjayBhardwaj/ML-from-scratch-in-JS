
// initializing data
const IrisX = tf.tensor(iris).slice([0,1],[100,2])

// creating one hot encoded y vector.
const IrisY = tf.tensor(Array(100).fill([1,0],0,50).fill([0,1],50));

// normalizing data
const mIrisX = normalizeData(IrisX);


// training our model on this dataset.
const model = new perceptron();
model.train({x: mIrisX, y: IrisY});

console.log('training complete.');

const fac = .02;

const decisionBoundary = [fac*model.getWeights().flatten().arraySync()[0], 
                          fac*model.getWeights().flatten().arraySync()[1]];


// testing our model.
// console.log('test')
// model.test(IrisX.slice([0,0], [1,-1])).print();

// console.log('accuracy:-')
// checkAccuracy(IrisX, IrisY, model.test).print();

function perceptronViz(){


    const fac = .02;

    const perceptronVizData = [
        {
            x : mIrisX.slice([0, 0],[50, -1]).slice([0, 0],[-1,  1]).flatten().arraySync(),
            y : mIrisX.slice([0, 0],[50, -1]).slice([0, 1],[-1, -1]).flatten().arraySync(),
            mode : 'markers',
            type : 'scatter',

        },
        {
            x : mIrisX.slice([50, 0],[-1, -1]).slice([0, 0],[-1,  1]).flatten().arraySync(),
            y : mIrisX.slice([50, 0],[-1, -1]).slice([0, 1],[-1, -1]).flatten().arraySync(),
            mode : 'markers',
            type : 'scatter',

        },
        // {
        //     x : [-decisionBoundary[0], decisionBoundary[1],],
        //     y : [-decisionBoundary[0], decisionBoundary[1],],
        //     mode : 'lines',
        //     type : 'scatter'

        // }
    ];

    Plotly.newPlot('perceptronViz', perceptronVizData, {title: 'Rosenblatt\'s perceptron' });

}

perceptronViz();



// generate meshgrid
const grid = meshGridRange(range={x: {min: -2.1, max: +2.1},y: {min: -2.1, max: +2.1}},division=20);

// calculate pNorm for each value of meshgrid

const pNormGrid = grid.map( (a) =>{
    const f = tf.tensor(a).transpose().arraySync();
    const w = f.map( (b) =>{ 
        // return 1*(pNorm(tf.tensor(b).expandDims(1), p=2).flatten().arraySync()[0] )
        return 1*model.test(tf.tensor(b).expandDims(1).transpose()).flatten().arraySync()[0];
        // return 1*(inducedMatrixNorm(tf.tensor([[1, 2],[0, 2]]),tf.tensor(b).expandDims(1),p=2).flatten().arraySync()[0] )
    });
    return w;
});

// visualzing the pNorm Grid
const pNormVizData = [{
    x : grid[0][0],
    y : grid[0][0],
    z : pNormGrid,
    type: 'contour',

    colorscale : [[0, darkModeCols.blue()], [0.25, darkModeCols.purple()],[0.5, darkModeCols.magenta()], [.75, darkModeCols.yellow()], [1, darkModeCols.red()]],

    contours: {
        start: -1,
        end: 1,
        size: 3
     },
    line : {
    },
}];

const layoutSetting = {
    title : 'p-Norm',
    font : {
        size : 15,
        color: 'white',
        family : 'Helvetica'
    },
    paper_bgcolor : '#222633',


}

Plotly.newPlot('decisionBoundaryViz',pNormVizData,layoutSetting);