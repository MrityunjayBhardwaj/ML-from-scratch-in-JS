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