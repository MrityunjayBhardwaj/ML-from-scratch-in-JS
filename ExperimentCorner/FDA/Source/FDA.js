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

        // covert the arrays to tf.tensor
        const data0 = dataSplit[0]; // data of class 0
        const data1 = dataSplit[1]; // data of class 1

        // calculate the mean of the matrix
        const data0Mean = tf.mean(data0);
        const data1Mean = tf.mean(data1);

        // calculate covariance matrix of both the class 
        const data0Cov = tf.matMul(data0,data0.transpose());
        const data1Cov = tf.matMul(data1,data1.transpose());

        // Calculating total Covariance
        const totalCov = tf.matMul(data0Cov.transpose(),data1Cov);
    
        // Calculating covariance b/w class means.
        const classMeanCov = tf.mul( tf.sub(data0Mean,data1Mean) , tf.sub(data0Mean,data1Mean).transpose() );

        // combining both within and between covariance
        const combinedCov = tf.mul(totalCov,classMeanCov);

        // calculating the eigen vectors of combined Covariance which will give us our 'w' 
        
        // because tfjs doesn't have eigen decomposition method we are going to be using ndjs
        const ndCombCov = tf2nd(combinedCov);

        // calculate eigen vecs and converting back to tf.tensor
        const {0 : eigenVecs , 1 : eigenVals} = nd.la.eigen(ndCombCov);

        // eigen vectors of combined covariance is going to be our 'w' which maximize our FDA objective function
        const w  = nd2tf(eigenVecs);

        // adding stuff to the model object
        model.weights = w;
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



