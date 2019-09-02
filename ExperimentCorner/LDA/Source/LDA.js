
function LDA(){

    const model = {
        params:{
            threshold: 0.5,
            mean: [],
            sharedCovariance: [],
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
            
            const currClassPrior = currClassDataLength / dataLength;
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

    },
    this.classify = (dataX, probOrClass=0, threshold)=>{
        threshold = threshold || model.params.threshold;
        // calculate the class conditional probabilities for all the classes.

        const nClasses = model.params.mean.length;

        const invCovariance = tfpinv(model.params.sharedCovariance);

        let posteriorProb = tf.tensor([]);

        for(let i=0;i< nClasses;i++){

            const currMean = model.params.mean[i];
            const currPrior = model.params.prior[i];

            const weights = invCovariance.matMul(currMean.transpose());
            const bias =  currMean.matMul(invCovariance).matMul( currMean.transpose() ).mul( -1/2 )
                            .add(
                            tf.log( currPrior) 
                            );


            const linearFn = weights.transpose().matMul(dataX.transpose()).add(bias);
            const currClassLogisticFn = this.logisticFn(linearFn);

            posteriorProb = posteriorProb.concat( (currClassLogisticFn) )

        }

        // converting posterior probabilites to classes 
        const predY = tf.tensor(1).sub(tf.abs( (posteriorProb.sub(tf.max( posteriorProb, axis=0 ))) ).mul(100000000).clipByValue(0,1)).transpose().matMul(tf.linspace(0,nClasses-1,nClasses).expandDims(1));
        
        if (probOrClass === 0)
            return posteriorProb; 
        
        return predY 
    

    }
}