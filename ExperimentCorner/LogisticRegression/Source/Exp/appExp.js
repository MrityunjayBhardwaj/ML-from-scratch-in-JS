
// initializing data
const mIrisX = tf.tensor(iris).slice([0,1],[100,2])
// one hot encoded
const mIrisY = tf.tensor(Array(100).fill([1,0],0,50).fill([0,1],50));

const model = new LogisticRegression();

const mIrisX_c0  = mIrisX.slice([0,0],[50,-1]);
const mIrisX_c1  = mIrisX.slice([50,0],[-1,-1]);

Plotly.newPlot('logisticRegressionViz');

function trainingCallback(x, y, yPred, Weights, Loss){

    console.log("inside Traning CallBack Function");
    const mIrisX_c0  = x.slice([0,0],[50,-1]);
    const mIrisX_c1  = x.slice([50,0],[-1,-1]);

    vizDecisionBoundary( dataX0=mIrisX_c0, dataX1=mIrisX_c1, classifierFn=model.logisticFn(.5,1,Weights).bind(model), divName='logisticRegressionViz', this );

    window.setTimeout(function(){console.log("times Up")},1000)
}

model.train( {x: mIrisX, y: mIrisY}, trainingCallback  );

console.log("trained");

vizDecisionBoundary( dataX0=mIrisX_c0, dataX1=mIrisX_c1, classifierFn=model.classify.bind(model), divName='logisticRegressionViz' );

function vizDecisionBoundary(dataX0,dataX1,classifierFn,divName='',bindTo=this,rangeX=null,rangeY=null,gridRes=50,margin=.3){
    // TODO: if the name is not given then just create one.

    // console.log("inside")
    const dataX = dataX0.concat(dataX1, axis=0);

    // console.log(dataX.shape)

    // initializing range/position for our grid Mesh
    let gridRange0 = [ dataX.min().flatten().arraySync()[0] - margin, dataX.max().flatten().arraySync()[0] + margin];
    let gridRange1 = [ dataX.min().flatten().arraySync()[0] - margin, dataX.max().flatten().arraySync()[0] + margin];

    if (rangeX && rangeY){
        gridRange0 = [rangeX[0]-margin,rangeX[1] + margin]
        gridRange1 = [rangeY[0]-margin,rangeY[1] + margin]
    }


    // creating grid Mesh
    const psudoPts0 = tf.linspace( gridRange0[0],gridRange0[1],gridRes).expandDims(1);
    const psudoPts1 = tf.linspace( gridRange1[0],gridRange1[1],gridRes).expandDims(1);
    const psudoPts  = psudoPts0.concat(psudoPts1,axis=1);
    const meshGridPsudoPts = meshGrid(psudoPts1.flatten().arraySync(),psudoPts0.flatten().arraySync());

    // classifying each point
    const meshGridPsudoPtsY = meshGridPsudoPts.map( 
            function(cRow) {
                // console.log(cRow);
            return classifierFn( tf.tensor(cRow).transpose() ).arraySync().map( (oneHotClass)=> oneHotClass.indexOf(1) )
            }
        );

    // Visualizing 

    const decisionRegionData = {
        x : psudoPts.slice([0,0],[-1,1]).flatten().arraySync(),
        y : psudoPts.slice([0,1],[-1,1]).flatten().arraySync(),
        z : meshGridPsudoPtsY,
        type: 'contour',
            colorscale:[[0, 'rgb(153, 153, 255)'],[1, 'rgb(255, 153, 102)']],
        contours : {
        // coloring : 'heatmap',
        // zsmooth: 'best',
        start: 0,
        end : 1,
        size: 2
        },
        line : {
        // width: .5,
            smoothing: 0
        },

    }

    const inputSpaceVizData = [{
        x: dataX0.slice([0,0],[-1,1]).flatten().arraySync(),
        y: dataX0.slice([0,1],[-1,-1]).flatten().arraySync(),

        mode: 'markers',
        type: 'scatter',

        marker : {
            width: 5
        }
    },
    {
        x: dataX1.slice([0,0],[-1,1]).flatten().arraySync(),
        y: dataX1.slice([0,1],[-1,-1]).flatten().arraySync(),

        mode: 'markers',
        type: 'scatter',

        marker : {
            width: 5
        }
    },
        decisionRegionData

    ]

    // console.log("plotting!",bindTo)


    bindTo.Plotly.newPlot(divName,inputSpaceVizData,{title: 'input Space'})












}

// EXP
// Visualizing Simoid prob.

// const gridMeshLogit = meshGridPsudoPts.map( 
//         function(cRow) {
//             // console.log(cRow);
//         return model.logisticFn(threshold=0.5,convert2Class=false)( tf.tensor(cRow).transpose(), model.getWeights() ).flatten().arraySync()
//         }
//     );

// const logitVizData = [{
//     x : psudoPts.slice([0,0],[-1,1]).flatten().arraySync(),
//     y : psudoPts.slice([0,1],[-1,1]).flatten().arraySync(),
//     z: gridMeshLogit,
//     type: 'contour',
//     line: {
//         width: 0
        
//     },
//     zsmooth: 'best',
//     contours: {
//         smooting : 1,
//         coloring: 'heatmap'
//     }
// }]

// Plotly.newPlot('sigmoidCurveViz',logitVizData,{title: 'Logit Function Viz'})




