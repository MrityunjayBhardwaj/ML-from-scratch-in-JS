
// initializing data
const mIrisX = tf.tensor(iris).slice([0,1],[100,2])
// one hot encoded
const mIrisY = tf.tensor(Array(100).fill([1,0],0,50).fill([0,1],50));

const model = new LogisticRegression();

const mIrisX_c0  = mIrisX.slice([0,0],[50,-1]);
const mIrisX_c1  = mIrisX.slice([50,0],[-1,-1]);

let weightHistory = [];

function trainingCallback(x, y, yPred, Weights, Loss){


    console.log("inside Traning CallBack Function");
    const mIrisX_c0  = x.slice([0,0],[50,-1]);
    const mIrisX_c1  = x.slice([50,0],[-1,-1]);

    weightHistory.push(Weights)
    // Weights.print();

    // vizDecisionBoundary( dataX0=mIrisX_c0, dataX1=mIrisX_c1, classifierFn=model.logisticFn(.5,1,Weights).bind(model), divName='logisticRegressionViz', window );

}

model.train( {x: mIrisX, y: mIrisY}, trainingCallback  );
weightHistory.push(model.getWeights())
console.log("trained");


const slider = document.getElementById('myRange');

let currWeight = model.getWeights();
slider.onchange = function(){
    
    const weightIndex = Math.floor( weightHistory.length*slider.value );
    const cWeight = weightHistory[weightIndex];
    currWeight = cWeight;
    vizDecisionBoundary( dataX0=mIrisX_c0, dataX1=mIrisX_c1, classifierFn=model.logisticFn(.5, 1, cWeight).bind(model), divName='logisticRegressionViz', window );
    
}









vizDecisionBoundary( dataX0=mIrisX_c0, dataX1=mIrisX_c1, classifierFn=model.classify.bind(model), divName='logisticRegressionViz' );

function vizDecisionBoundary(dataX0,dataX1,classifierFn,divName='',bindTo=this,rangeX=null,rangeY=null,gridRes=50,margin=.3){
    // TODO: if the divName is not given then just create one.

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
                const cRowClass = classifierFn( tf.tensor(cRow).transpose() );

                // console.log("printing cRowClass");
                // cRowClass.print();
                if (cRowClass.shape[1] === 1)
                    return cRowClass.flatten().arraySync();

            return cRowClass.arraySync().map( (oneHotClass)=> oneHotClass.indexOf(0) );
            }
        );

    // console.log( meshGridPsudoP  tsY );

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

    console.log("plotting!")


    window.Plotly.newPlot(document.getElementById(divName),inputSpaceVizData,{title: 'input Space'})













// EXP
// Visualizing Simoid prob.

const gridMeshLogit = meshGridPsudoPts.map( 
        function(cRow) {
            // console.log(cRow);
        return model.logisticFn(threshold=0.5,convert2Class=false)( tf.tensor(cRow).transpose(), currWeight ).flatten().arraySync()
        }
    );

    console.log(gridMeshLogit,model.getWeights().print())

const logitVizData = [{
    x : psudoPts.slice([0,0],[-1,1]).flatten().arraySync(),
    y : psudoPts.slice([0,1],[-1,1]).flatten().arraySync(),
    z: gridMeshLogit,
    type: 'surface',
    line: {
        width: 0
        
    },
    zsmooth: 'best',
    contours: {
        smooting : 1,
        coloring: 'heatmap'
    }
},

{
    x : dataX0.slice([0,0],[-1,1]).flatten().arraySync(),
    y : dataX0.slice([0,1],[-1,-1]).flatten().arraySync(),
    z : tf.zeros([dataX0.shape[0],1]).flatten().arraySync(),
    mode: 'markers',
    type: 'scatter3d'
},
{
    x : dataX1.slice([0,0],[-1,1]).flatten().arraySync(),
    y : dataX1.slice([0,1],[-1,-1]).flatten().arraySync(),
    z : tf.zeros([dataX0.shape[0],1]).flatten().arraySync(),
    mode: 'markers',
    type: 'scatter3d'
}
]

// const logitViz2D = [{
//     x : psudoPts1.flatten().arraySync(),
//     y : tf.matMul(psudoPts, currWeight).flatten().arraySync(),

//     type: 'scatter'
// }]


Plotly.newPlot('sigmoidCurveViz',logitVizData,{title: 'Logit Function Viz'})





}