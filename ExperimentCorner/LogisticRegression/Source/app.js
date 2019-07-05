
// initializing data
const mIrisX = tf.tensor(iris).slice([0,1],[100,2])
// one hot encoded
const mIrisY = tf.tensor(Array(100).fill([1,0],0,50).fill([0,1],50));

const model = new LogisticRegression();
model.train({x: mIrisX, y: mIrisY});


const psudoPtsGridRes = 50;
const margin = .3;
let gridRange0 = [ mIrisX.min().flatten().arraySync()[0] - margin, mIrisX.max().flatten().arraySync()[0] + margin];
let gridRange1 = [ mIrisX.min().flatten().arraySync()[0] - margin, mIrisX.max().flatten().arraySync()[0] + margin];
// gridRange0[0] = 3;
// gridRange1[0] = 2;
gridRange0 = [1-margin,5 + margin]
gridRange1 = [0-margin,6 + margin]


const psudoPts0 = tf.linspace( gridRange0[0],gridRange0[1],psudoPtsGridRes).expandDims(1);
const psudoPts1 = tf.linspace( gridRange1[0],gridRange1[1],psudoPtsGridRes).expandDims(1);
const psudoPts  = psudoPts0.concat(psudoPts1,axis=1);

const meshGridPsudoPts = meshGrid(psudoPts1.flatten().arraySync(),psudoPts0.flatten().arraySync());
const meshGridPsudoPtsY = meshGridPsudoPts.map( 
        function(cRow) {
            // console.log(cRow);
        return model.classify( tf.tensor(cRow).transpose() ).arraySync().map( (oneHotClass)=> oneHotClass.indexOf(1) )
        }
    );

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
    x: mIrisX.slice([0,0],[50,-1]).slice([0,0],[-1,1]).flatten().arraySync(),
    y: mIrisX.slice([0,0],[50,-1]).slice([0,1],[-1,-1]).flatten().arraySync(),

    mode: 'markers',
    type: 'scatter',

    marker : {
        width: 5
    }
},
{
    x: mIrisX.slice([50,0],[-1,-1]).slice([0,0],[-1,1]).flatten().arraySync(),
    y: mIrisX.slice([50,0],[-1,-1]).slice([0,1],[-1,-1]).flatten().arraySync(),

    mode: 'markers',
    type: 'scatter',

    marker : {
        width: 5
    }
},
decisionRegionData

]

Plotly.newPlot('logisticRegressionViz',inputSpaceVizData,{title: 'input Space'})



// EXP
// Visualizing Simoid prob.

const gridMeshLogit = meshGridPsudoPts.map( 
        function(cRow) {
            // console.log(cRow);
        return model.logisticFn(threshold=0.5,convert2Class=false)( tf.tensor(cRow).transpose(), model.getWeights() ).flatten().arraySync()
        }
    );

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
}]

Plotly.newPlot('sigmoidCurveViz',logitVizData,{title: 'Logit Function Viz'})




