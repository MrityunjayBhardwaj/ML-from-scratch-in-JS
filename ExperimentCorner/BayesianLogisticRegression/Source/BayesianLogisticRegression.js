function bayesianLogisticRegression(){

    const model = {
        predictiveProb : 0,
        parameterPDF: {
            mean : [],
            covariance: []
        }
    }

    this.logisticFn = function(a){
        return tf.exp(tf.neg(a)).add(1).pow(-1);
    }

    this.parameterPDF = function(data){

        const nClasses = data.y.shape[1];
        const dataSplit = classwiseDataSplit(data.x, data.y,concatClass=1);

        // hyperparameters for prior distribution.
        const hypParams = { m_0: tf.zeros([data.x.shape[1]]), 
                            S_0: tf.ones([nClasses,nClasses]) };


        // TODO: Calculate the W_MAP For each Classes>>>>


        for(let i=0; i<nClasses; i++){

            // let 1 for class_i and 0 for the rest

            const currClassDataX = dataSplit[i].x;

            const currY = tf.tensor(1);// specify the current class of the data.


            // const currWeightMAP = hypParams.m_0.sub( S_0.mul(tf.sum(currClassDataX - currY.mul(currClassDataX))));
    
            // from the second calculation.
            const currWeightMAP = tf.sum(currClassDataX, axis=0).pow(-1).expandDims(1);

            const predY = this.logisticFn( currClassDataX.matMul(currWeightMAP) );
            const currWeightCovariance = hypParams.S_0.add(
            tf.sum(
                predY.mul(
                tf.tensor(1)
                    .sub(predY)
                    .mul(currClassDataX.matMul(currClassDataX.transpose()))
                )
            )
            );

            model.parameterPDF.mean.push( currWeightMAP );
            model.parameterPDF.covariance.push( currWeightCovariance ); 

        }



        // TODO: Create a function to generate weights from the parameterPDF.
        // TODO: Create a function to output the probability of of given input weight.

    }
    this.train = function(data){
        const lambda = Math.PI/8; // parameter for inverse probit function.

        const dataX = data.x;

        const maxProb = 0;
        const optimalClass = -1;

        const nClasses = data.y.shape[1];

        // calculating the parameters of posterior probability
        this.parameterPDF(data);

        // calculating the posterior predictive probability.
        for(let i=0; i<nClasses; i++){

            const currMeanWeight = model.parameterPDF.mean[i]; // more presicely its MAP weight.
            const currWeightCovariance = model.parameterPDF.covariance[i]; 

            const currMean = dataX.matMul( currMeanWeight );
            const currVariance = currWeightCovariance.matMul(dataX.transpose() ).matMul( dataX );

            const kai = currVariance.mul(lambda).add(1).pow(-1/2);

            const currClassPredictiveProb = this.logisticFn( kai.mul(currMean) );

            if (maxProb < currClassPredictiveProb){ 
                maxProb = currClassPredictiveProb
                optimalClass = i;
            }
            model.predictiveProb.push( currClassPredictiveProb );
        }

        // return the class which has max posterior predictive probability

        // TODO: return the one hot encoded vector.
        return optimalClass
    }

    this.test = function(){

        const optimalClass = -1;

        const nClasses = data.y.shape[1];
        
        // calculating the posterior predictive probability.
        for(let i=0; i<nClasses; i++){

            const currClassPredictiveProb = this.logisticFn( kai.mul(currMean) );

            if (maxProb < currClassPredictiveProb){ 
                maxProb = currClassPredictiveProb
                optimalClass = i;
            }
            model.predictiveProb.push( currClassPredictiveProb );
        }
    }
}