
function QDA(){

    const model = {
        params:{
            threshold: 0.5,
            mean: [],
            sharedCovariance: [],
            covariance: [],
            prior: [],
        }
    }

    this.getParams= () =>{
        return model.params;
    }
    this.logisticFn = (z) =>{
        return  tf.exp(tf.neg(z)).add(1).pow(-1);
    }
    this.train = (data={})=>{


        // do classwise split
        const classwiseData = classwiseDataSplit(dataX=data.x, dataY=data.y, concatClass=1);


        // calcuate mean, covariance and prior probabilities of each of the class.
        // using MLE results

        const classMean = [];
        const classCovariance = [];
        const classPrior = [];

        const dataLength =  data.x.shape[0];  
        let sharedCovariance = tf.tensor([]);

        for(let i=0;i<classwiseData.length;i++){
            const currClassData = classwiseData[i].x;
            const currClassDataLength = currClassData.shape[0];
            
            const currClassPrior = tf.tensor( currClassDataLength / dataLength );
            const currClassMean  = tf.sum(currClassData, axis=0).div(currClassDataLength).expandDims(1).transpose();

            const meanCenteredCurrClassData = currClassData.sub(currClassMean);
            const currClassCovar = meanCenteredCurrClassData.transpose().matMul( meanCenteredCurrClassData ).div(currClassDataLength);
            
            if (sharedCovariance.shape[0]){
                sharedCovariance = sharedCovariance.add(currClassCovar.mul(currClassDataLength / dataLength));
            }
            else{
                sharedCovariance = (currClassCovar.mul(currClassDataLength / dataLength));
            }

            classPrior.push( currClassPrior );
            classMean.push( currClassMean );
            classCovariance.push( currClassCovar );
        }

        // inserting the calcuated parameters to our model object for further use.
        model.params.mean = classMean;
        model.params.sharedCovariance = sharedCovariance;
        model.params.prior = classPrior;
        model.params.covariance = classCovariance;

    },
    this.classify = (dataX, probOrClass=1, threshold)=>{
        // TODO: derived for individual covariance instead of common covariance.

        threshold = threshold || model.params.threshold;
        // calculate the class conditional probabilities for all the classes.
        const nClasses = model.params.mean.length;

        const c1Mean = model.params.mean[0];
        const c2Mean = model.params.mean[1];

        const c1Prior = model.params.prior[0];
        const c2Prior = model.params.prior[1];

        // const invCovariance = tfpinv(model.params.sharedCovariance);

        const c1InvCovariance = tfpinv(model.params.covariance[0]);
        const c2InvCovariance = tfpinv(model.params.covariance[1]);

        const c1DetCovariance = tfdet(model.params.covariance[0]);
        const c2DetCovariance = tfdet(model.params.covariance[1]);


        let posteriorProb = tf.tensor([]);

        const quadraticFn =  /* weights */
                          dataX.sub(c1Mean).matMul(c1InvCovariance).matMul( dataX.sub(c1Mean).transpose() )
                          .sub( dataX.sub(c2Mean).matMul(c2InvCovariance).matMul( dataX.sub(c2Mean).transpose() ) ).mul(1/2)

                          /* bias */
                          .add( tf.log( c1DetCovariance.div(c2DetCovariance) ).mul(1/2))
                          .add( tf.log( c1Prior.div(c2Prior) ));


    
        for(let i=0;i< nClasses;i++){

            const currMean = model.params.mean[i];
            const currInvCovariance = tfpinv(model.params.covariance[i]);

            // TODO: tfdet need calculate nxn determinant 

            const currDetCovariance = tfdet(model.params.covariance[i]);
            const currPrior = model.params.prior[i];

            const currClassQuadraticFn = dataX.sub(currMean).matMul(currInvCovariance).matMul( dataX.sub(currMean).transpose() ).mul(-1/2)
                                .sub( tf.log(currDetCovariance).mul(1/2))
                                .add(tf.log(currPrior));

            const currClassLogisticFn = this.logisticFn(currClassQuadraticFn);

            posteriorProb = posteriorProb.concat( currClassLogisticFn.mul(currPrior ));

        }

        const predY = tf.floor( tf.tensor(1).sub( tf.abs( (posteriorProb.sub(tf.max( posteriorProb, axis=0 ))) ).mul(1000000000).clipByValue(0,1) ).transpose().matMul(tf.linspace(0,nClasses-1,nClasses).expandDims(1)) );


        // const quadraticFn2 = dataX.sub(c1Mean).matMul(c1InvCovariance).matMul( dataX.sub(c1Mean).transpose() ).mul(-1/2)
        //                      .sub( tf.log(c1DetCovariance).mul(1/2))
        //                      .add(tf.log(c1Prior));

        // feeding our linear function to logistic sigmoid function
        // const classConditionalProb   = this.logisticFn(quadraticFn2);
        // const ccp   = this.logisticFn(quadraticFn2);

        

        // TODO: converting class conditional probabilities into classes 
        
        if (probOrClass === 0)
            return posteriorProb
        
        return predY;

    }
}