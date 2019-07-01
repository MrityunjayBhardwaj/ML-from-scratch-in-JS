
let matrixX = tf.tensor([[1,2],[2,5],[7,8],[12,15]]);

const weights = tf.tensor([3,2,5]).expandDims(1);

console.log(matrixX.print())

console.log(matrixX.print())
const matrixXMod = matrixX.concat( tf.ones([matrixX.shape[0],1]), axis=1);
let matrixY = matrixXMod.matMul( weights );

matrixY.print()
// adding some noise
matrixY = tf.add(matrixY, tf.randomNormal(matrixY.shape,0,10));

matrixY.print();


function addNDrag(){

}


function vizLeastSquares(matrix){

    const model = new LeastSquares();
    const leastSquareFitting = model.train( matrix );

    const k =[0,30];
    const range = [ [ -k[0], -k[0] ], 
                    [  k[1], -k[0] ],
                    [ -k[0],  k[1] ],
                    [  k[1],  k[1] ], ];
    
    const surf = tf.tensor(range);


    const lsRange = model.test( surf ) ;

    const ls = lsRange.flatten().arraySync();

    console.log(lsRange.flatten().arraySync(), ls[0])

    // const pusodX =
    const lsVizData = [
    {
        x: matrix.x.transpose().arraySync()[0],
        y: matrix.x.transpose().arraySync()[1],
        z: matrix.y.transpose().arraySync()[0], 
        mode: 'markers',
        type: 'scatter3d',
    },

    {
        x : surf.slice([0,0],[-1,1]).flatten().arraySync(),
        y : surf.slice([0,1],[-1,1]).flatten().arraySync(),
        // z : lsRange.reshape([2,2]).reverse().arraySync(),
        z: [[ls[0],ls[1]],[ls[0],ls[1]],
            [ls[2],ls[3]],[ls[2],ls[3]]],
        type: 'surface',
        opacity: 0.8,
        colorscale:[[0,'red'],[1,'red']]
    }
    
    ]

    console.log(lsVizData)

    Plotly.newPlot('lsViz',lsVizData,{title:"Input Data"});
}

vizLeastSquares({x: matrixX, y: matrixY});