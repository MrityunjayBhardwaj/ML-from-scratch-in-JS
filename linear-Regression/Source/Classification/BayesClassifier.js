
/**
 * 
 * @param {tf.tensor} x tf.tensor : X
 * @param {tf.scalar} mean tf.scalar : mean of Gaussian PDF
 * @param {tf.scalar} variance tf.scalar : variance of gaussian PDF
 */
function gaussianPDF(mean,variance){

    this.calc =  (x) =>  tf.mul( tf.scalar( 1 / ( Math.sqrt(2*Math.PI*variance.arraySync()) ) ) , 
                    tf.exp( 
                            tf.mul( tf.scalar(1/2) ,
                            tf.pow( tf.sub( x , mean ) , 2 )
                                  )
                           ) 
                  ) 
}


function calcCoverianceMatrix(x,xMean){

    const nSamples = x.shape[0];
    const stdev =  tf.add( tf.sub( x , xMean ) , axis=1 ).arraySync();
    
    // covariance matrix:-
    const covMatrix = [];
    for(let i=0;i<nClasses ;i++){
        
        const covarianceRow = [];
        for(let j=0;j<nClasses ;j++){

            const currCovariance = ( stdev[i]*stdev[j] ) / (nSamples - 1);
            covarianceRow.push(currCovariance);
        }
        covMatrix.push(covarianceRow);
    }

    return covMatrix;
    
}

function calcDeterminant(cov){
    // NOTE: currently supports 2x2 det only.
    cov = cov.arraySync();
    const a = cov[0][0], b = cov[0][1] ,
          c = cov[1][0], d = cov[1][1]
    return tf.scalar( a*d - b*c );
}

/**
 * 
 * @param {tf.tensor} x Training Data feature vector
 * @param {tf.tensor} y one hot encoded output vector
 * @param {function} likelihoodPDF collection of function of all the feature vector/cols in data , if not specified, its assumed to be a gaussian.
 */
function BayesClassifier(x,y){

    // private functions and variables:-
    const nClasses = x.shape[1];
    const nSamples = x.shape[0];

    const likelihoodFnList = [] ;

    function classwiseDataSplit(x,y){
        // make sure y is a one hot encoded vector.


        const yArray = y.flatten().arraySync();
        const xArray = x.arraySync();

        const xSplit = []
        for(let i=0;i<nClasses;i++){
            Xsplit[i] = xArray.filter(function(_,index){ return this[index][i]; },yArray);
            Xsplit[i] = tf.tensor( Xsplit[i] );
        }

        return xSplit; 
        
    }
    this.train = function(){

        // calculate prior
        // calculate likelihood prob.
        // use bayes rule

        /**
         * 1 ) split the data class wise DONE!
         * 2 ) calculate the prior and likelihood prob for each classes
         * 3 ) test the new data points  
         * optional : construct the decision boundry.
         */

         // classwise data spliting
        const xClassSplit =  classwiseDataSplit(x,y);

        // calculate the prior and likelihood fns.
        for(let i=0;i<nClasses;i++){
           
            // calculate the curr params for this particular class
            const currX = xClassSplit[i];

            const currMean =  tf.mean(currX,axis=1);
            const currVariance = calcCoverianceMatrix(currX,currMean);

            // caluclate and feed this new updated likelihood function;
            const currLikelihoodFn = new gaussianPDF(currMean,currVariance);

            likelihoodFnList.push(currLikelihoodFn);
        }

    }
    this.test = function(testX){
        // calculate and return all the probability from all the classes.

        const results = [];
        for(let i=0;i<nClasses;i++){
            const cFn = likelihoodFnList[i];
            results.push( cFn().calc(testX).arraySync() );
        }
        return results;
    }
}
