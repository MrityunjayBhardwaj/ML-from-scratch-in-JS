/**
 * 
 * @param {tf.tensor} matrix 2x2 matrix 
 * @param {object} range format: {min : Number , max : Number} ; specify the range of the quad surface to be calculated
 * @param {boolean} isNormalized should we calculate the normalized quadratic form?
 * @param {Number} res specify the resoultion of our quadratic surface
 * @description given a tf.tensor this function produce the quadratic surface. of the provided range
 */
function quadForm(inputMatrix,range={min: 0, max: 1},isNormalized,res = 100){

    // creating input grid pts for our quadSurf
    const gridAxisPts = tf.linspace(range.min,range.max,res);
    let gridMesh    = meshGrid(gridAxisPts.arraySync(),gridAxisPts.arraySync());

    // Using Tensor Mathematics :- calculating W^T * S * W
    const inputMtxDim = inputMatrix.shape;

    /* preparing for matrix Multiplication */
    let inputMtxTensor = inputMatrix; // our 'S'
    inputMtxTensor = inputMtxTensor.reshape( [1,inputMtxDim[0],inputMtxDim[1] ] ); // [1,2,2]

    // tiling the inputMatrix in order to matmul for each row in gm
    inputMtxTensor = inputMtxTensor.tile([res,1,1]).expandDims(1); /// "3" => no. of rows in gm and also res = 3


    let gridMeshTensor = tf.tensor( gridMesh ); // our 'W'
    gridMeshTensor = gridMeshTensor.reshape([res, 1, inputMtxDim[1], res]).transpose([0,1,3,2]); // [3,1,2,3] and [0,1,3,2]

    // calculating sol = W^T * S
    let sol = gridMeshTensor.matMul(inputMtxTensor);

    // preparing for final matrix multiplication 
    sol = sol.reshape([res,res,1,inputMtxDim[1]]); // [3,3,1,2]
    gridMeshTensor = gridMeshTensor.reshape([res,res,inputMtxDim[1],1]); // [3,3,2,1]

    // finally calculating sol* W
    let finalSolution = sol.matMul(gridMeshTensor).reshape([res,res]);


    if (isNormalized){

        // calculating the normalization factor : W^T * W
        const gridMeshTransposed = (gridMeshTensor.reshape([res,res,1,inputMtxDim[1]]));
        const normFac = gridMeshTransposed.matMul(gridMeshTensor).reshape([res,res]);
        
        finalSolution = tf.div(finalSolution, normFac);
    }

    return { 
             x: gridAxisPts.flatten().arraySync(),
             y: gridAxisPts.flatten().arraySync(),
             z: finalSolution.arraySync(),
            };
}

function quadFormUsingMap(matrix,range={min: 0, max: 1},isNormalized,res = 100){

    // creating input grid pts for our quadSurf
    const gridAxisPts = tf.linspace(range.min,range.max,res);
    let gridMesh    = meshGrid(gridAxisPts.arraySync(),gridAxisPts.arraySync());
    const w = gridAxisPts;

    // calculating and returning Quadrtic Surface:-
    const quadSufMtx = gridMesh.map( function(cRow,i){
        cRow = tf.tensor(cRow).transpose().arraySync()

        const calcVal =   cRow.map( function(x){

            // calculating quadraatic Surface
            const tfX = tf.tensor(x).expandDims(1);
            let quadSurf = tf.matMul ( tfX.transpose(), matrix).matMul(tfX);

            // normalize the quadSuf if specified
            if (isNormalized){
                const normFac = tf.matMul( tfX.transpose() , tfX); 
                quadSurf = tf.div( quadSurf , normFac);
            }
            
            return quadSurf.flatten().arraySync()[0];
        } )

        return calcVal;

    } );
    return { x: gridAxisPts.flatten().arraySync(), y: gridAxisPts.flatten().arraySync(), z: quadSufMtx };

}