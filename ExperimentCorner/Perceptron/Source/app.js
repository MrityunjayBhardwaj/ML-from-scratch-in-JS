
// initializing data
const IrisX = tf.tensor(iris).slice([0,1],[100,2])

// creating one hot encoded y vector.
const IrisY = tf.tensor(Array(100).fill([1,0],0,50).fill([0,1],50));

// normalizing data
const mIrisX = normalizeData(IrisX);

// training our model on this dataset.
const model = new perceptron();
model.train({x: mIrisX, y: IrisY});

function perceptronViz(){


    const fac = .02;

    const perceptronVizData = [{
        x : mIrisX.slice([0, -1],[50, -1]).slice([0, 0],[-1,  1]).flatten().arraySync(),
        y : mIrisX.slice([0, -1],[50, -1]).slice([0, 1],[-1, -1]).flatten().arraySync(),
        mode : 'markers',
        type : 'scatter',

    },
    {
        x : mIrisX.slice([50, -1],[-1, -1]).slice([0, 0],[-1,  1]).flatten().arraySync(),
        y : mIrisX.slice([50, -1],[-1, -1]).slice([0, 1],[-1, -1]).flatten().arraySync(),
        mode : 'markers',
        type : 'scatter',

    },
    {
        x : [-fac*model.getWeights().flatten().arraySync()[0], fac*model.getWeights().flatten().arraySync()[0]],
        y : [-fac*model.getWeights().flatten().arraySync()[1], fac*model.getWeights().flatten().arraySync()[1]],
        mode : 'lines',
        type : 'scatter'

    }



    ];

    Plotly.newPlot('perceptronViz', perceptronVizData, {title: 'Rosenblatt\'s perceptron' });

}

perceptronViz();