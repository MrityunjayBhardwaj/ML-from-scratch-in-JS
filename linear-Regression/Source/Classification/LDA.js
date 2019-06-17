// it uses Bayes classifier but here, we use a common covariance matrix among all the classes and different
// mean.


/**
 * 
 * @param {tf.tensor} x tf.tensor : X
 * @param {tf.scalar} mean tf.scalar : mean of Gaussian PDF
 * @param {tf.scalar} variance tf.scalar : variance of gaussian PDF
 */
function gaussianPDF(mean,variance){

    let mu = mean;
    let sigma = variance;
    this.check = function(){
        mu.print();
        sigma.print();
        console.log(mu,sigma);
    }
    this.calc =  function(x){ 

        let k = tf.mul( tf.scalar( 1 / ( Math.sqrt(2*Math.PI*variance.arraySync()) ) ) , 
                    tf.exp( 
                            tf.mul( tf.scalar(1/2) ,
                            tf.pow( tf.sub( x , mean ) , 2 )
                                  )
                           ) 
                  ) 
                  
                return k;
                }
}

/**
 * 
 * @param {*} x 
 * @param {*} xMean 
 * @description Calcuate the Covariance Matrix
 */
function calcUniCoverianceMatrix(x,xMean){

    const nSamples  = x.shape[0];
    const nFeatures = x.shape[1];
   
    const stdev =  tf.sub( x , xMean );
    
    // covariance matrix:-
    const covMatrix = [];
    for(let i=0;i<nFeatures;i++){
        
        const covarianceRow = [];
        for(let j=0;j< nFeatures;j++){

            const splitA = stdev.slice([0,i],[-1,1]);
            const splitB = stdev.slice([0,j],[-1,1]);

            const currCovariance = tf.div( tf.sum( tf.mul(splitA,splitB),axis=0 ) , nSamples-1 );
            covarianceRow.push(currCovariance.arraySync()[0]);
        }
        covMatrix.push(covarianceRow);
    }

    // console.log(covMatrix);
    return tf.tensor(covMatrix);
    
}

/**
 * 
 * @param {tf.tensor} cov 
 * @description NOTE: Currently it only supports the 2x2 matrix.
 */
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

    const likeNPriorList = [] ;

    function classwiseDataSplit(x,y){
        // make sure y is a one hot encoded vector.


        const yArray = y.arraySync();
        const xArray = x.arraySync();

        const xSplit = []
        for(let i=0;i<nClasses;i++){
            let currClassSplit = xArray.filter(function(_,index){ return this[index][i]; },yArray);
            xSplit.push( tf.tensor( currClassSplit ) );
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

            const currMean =  tf.mean(currX,axis=0);
            const currVariance = calcCoverianceMatrix(currX,currMean);
            const varDet = calcDeterminant(currVariance);
            
            console.log("yey")
            varDet.print();
            // caluclate and feed this new updated likelihood function;
            const currLikelihoodFn = new gaussianPDF(currMean,varDet);
            const currPriorProb    = currX.shape[0]/nSamples;

            likeNPriorList.push({likelihoodFn: currLikelihoodFn,priorProb: currPriorProb});
        }

    }
    this.test = function(testX){
        // calculate and return all the probability from all the classes.

        const results = [];
        for(let i=0;i<nClasses;i++){
            const cFn = likeNPriorList[i].likelihoodFn;
            const likelihoodProb = cFn.calc(testX).arraySync();
            const priorProb = likeNPriorList[i].priorProb; /* TODO: Calculate the ratio */;

            const posteriorProb = likelihoodProb*priorProb;  // TODO: we also have to normalize it.
            results.push( posteriorProb );
        }
        console.log(results);
        return results;
    }
}
