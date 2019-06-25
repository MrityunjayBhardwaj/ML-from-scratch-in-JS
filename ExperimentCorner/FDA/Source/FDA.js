/**
 * 1 ) calculate S_w and S_b  
 *  where, S_b = covariance_0^Tcovariance_1 = total cov
 *  S_w = (mu_0 - mu_1)*(mu_0 - mu_1)^T = UU^T = covariance b/w the mean of the 2 classes
 * 
 * 2 ) eigen Decomp of (S_w*S_b) which will give us the w i.e, eigenvectors
 */

function FDA(data){

    const model = {
        weights : null,
    }

    this.data = data;
    this.train = function(data){

        const tfDataX = tf.tensor(data.x);
        const tfDataY = tf.tensor(data.y);

        // do classwise division 
        const dataSplit = classwiseDataSplit(tfDataX,tfDataY);

        const data0 = dataSplit[0]; // data of class 0
        const data1 = dataSplit[1]; // data of class 1

        // calculate the mean of the matrix
        const data0Mean = tf.mean( data0 , axis=0 );
        const data1Mean = tf.mean( data1 , axis=0 );

        // calculate covariance matrix of both the class 
        const data0Cov = tf.matMul(data0.transpose(),data0);
        const data1Cov = tf.matMul(data1.transpose(),data1);

        // Calculating total Covariance
        const totalCov = tf.matMul(data0Cov.transpose(),data1Cov);
    
        // Sw_inv * ( mu0 - mu1)
        const tCovInv = tf.tensor( pinv( totalCov.arraySync() ) );

        const s = tf.matMul( tCovInv, tf.sub( data0Mean, data1Mean ).expandDims(1) );

        s.print();

        // adding stuff to the model object
        model.weights = s;

        return s;
    }
    this.test = function(testX){
        
        // convert to tensor
        testX = tf.tensor(testX);
        return tf.matMul(model.weights.transpose(),testX);
    }

    /**
     * Visualize the components of FDA
     */
    this.viz = function(){
        
    }
}




function FDAmc(){

    // TODO: account for the case when N_k of all the classes are unequal.

    this.train = function(data){
        // TODO: make the input to always be a tf.tensor object
        const tfDataX = tf.tensor(data.x);
        const tfDataY = tf.tensor(data.y || []);

        // do classwise split:
        const dataClassArray = classwiseDataSplit(tfDataX,tfDataY);

        // calculating the mean of all the classes.
        const dataMean = dataClassArray.map( (cClassData) => tf.mean( cClassData, axis=0 ).expandDims(1) );       

        // calculate the covariance matrix for all the classes data
        const dataCov = dataClassArray.map( (cClassData) => tf.matMul( cClassData.transpose(),cClassData ) );

        // Cummulative mean of all the data points from all the classes
        const cumMean = tf.mean( tfDataX, axis=0).expandDims(1);

        // let sB = dataMean.map( (cMean) => tf.pow( tf.sub(cumMean , cMean), 2 ) );
        // sB = tf.concat( sB, axis=1 );

        const dataSD = tf.sub( tfDataX, cumMean.transpose() );

        const sT = tf.mul(tf.matMul( dataSD.transpose(), dataSD ), tf.scalar(tfDataX.shape[0]));

        // calculating total covariance:-
        const sW = tf.addN(dataCov);
        const sW_inv = pinv( sW.arraySync() );

        const sB = tf.sub( sT, sW);

        // combining sW and sB
        const matrixA = tf.matMul( sW_inv, sB );

        // const aSVD = nd.la.svd_decomp( matrixA.arraySync() );
        // calculating eigenVectors for each matrix:-
        const {0 : eigenVals , 1 : eigenVecs} = nd.la.eigen( tf2nd(matrixA) );
        // const {0 : lsVec , 1 : singularVals, 2: rsVec } = aSVD ;

        console.log( eigenVals)
        return convert2dArray(eigenVecs);
        // console.log(eigenVals,eigenVecs);
    }
}