/**
 * 
 * @param {tf.tensor} matrix 2x2 matrix 
 * @param {object} range format: {min : Number , max : Number} ; specify the range of the quad surface to be calculated
 * @param {boolean} isNormalized should we calculate the normalized quadratic form?
 * @param {Number} res specify the resoultion of our quadratic surface
 * @description given a tf.tensor this function produce the quadratic surface. of the provided range
 */
function quadForm(matrix,range={min: 0, max: 1},isNormalized,res = 100){

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
            if (0){
                const normFac = tf.matMul( tfX.transpose() , tfX); 
                quadSurf = tf.div( quadSurf , normFac);
            }
            
            return quadSurf.flatten().arraySync()[0];
        } )

        return calcVal;

    } );

    console.log(quadSufMtx)

    // Using Tensor Mathematics :-
    let modMtx4 = matrix;
    modMtx4 = modMtx4.reshape( [1,matrix.shape[0],matrix.shape[1]] ); // [1,2,2]

    // tiling the matrix in order to matmul for each row in gm
    modMtx4 = modMtx4.tile([res,1,1]).expandDims(1); /// "3" => no. of rows in gm and also res = 3

    let gm4 = tf.tensor( gridMesh );

    // Note :- this only works for 2d values.
    gm4 = gm4.reshape([res, 1, 2, res]).transpose([0,1,3,2]); // [3,1,2,3] and [0,1,3,2]

    // console.log("gm4: ");
    // gm4.print();

    // console.log("modMtx4: ");
    // modMtx4.print();

    let sol = gm4.matMul(modMtx4);
    sol = sol.reshape([3,3,1,2]);

    gm4 = gm4.reshape([3,3,2,1]);

    let finalSolution = sol.matMul(gm4).reshape([3,3]);

    finalSolution.print(); 



    return { x: gridAxisPts.flatten().arraySync(), y: gridAxisPts.flatten().arraySync(), z: finalSolution.arraySync() };
}