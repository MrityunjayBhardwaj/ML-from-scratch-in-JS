// Principle Component analysis

function PCA(){
    const model = {};

    function calcEigen(covMtx){
        // converting to nd array object.
       const ndCovMtx =  nd.array(covMtx.arraySync() );

       // converting back to tf.tensor
       return nd.la.eigen(ndCovMtx);

    };
    this.train = function(x,y){
        // Calcuate the covariance Matrix
        const covMtx = calcCoverianceMatrix(x);
        const {eigenvalues,eigenvectors} = calcEigen(covMtx);
    }
    this.test = function(testX){
        // take the first principle component and do something with it.
    }
}