
function LDA(){

    model = {
        params:{
            threshold: 0.5,
            mean: [],
            sharedCovariance: [],
            prior: [],
        }
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

        const dataLength =  data.shape[0];  
        let sharedCovariance = tf.tensor([]);

        for(let i=0;i<classwiseData.length;i++){
            const currClassData = classwiseData[i];
            const currClassDataLength = currClassData.shape[0];
            
            const currClassPrior = currClassDataLength / dataLength;
            const currClassMean  = tf.sum(currClassData).div(currClassDataLength);

            const meanCenteredCurrClassData = currClassData.sub(currClassMean);
            const currClassCovar = meanCenteredCurrClassData.matMul( meanCenteredCurrClassData.transpose() ).div(currClassDataLength);
            
            sharedCovariance = sharedCovariance.add(currClassCovar.mul(currClassDataLength / dataLength));

            classPrior.push( currClassPrior );
            classMean.push( currClassMean );
            classCovariance.push( currClassCovar );
        }

        // inserting the calcuated parameters to our model object for further use.
        model.params.mean = classMean;
        model.params.sharedCovariance = sharedCovariance;
        model.params.prior = classPrior;

    },
    this.classify = (data, probOrClass=0, threshold)=>{
        threshold = threshold || model.params.threshold;
        // calculate the class conditional probabilities for all the classes.

        const c1Mean = model.params.mean[0];
        const c2Mean = model.params.mean[1];

        const bayesFactor = model.params.prior[0] / model.params.prior[1];

        const invCovariance = tfpinv(model.params.sharedCovariance);
        const weights = tfpinv(model.params.sharedCovariance).mul(c1Mean.sub(c2Mean));
        const bias =  c1Mean.transpose().matMul(invCovariance).matMul(c1Mean).mul( -1/2 )
                        .add(
                        c2Mean.transpose().matMul(invCovariance).matMul(c2Mean).mul( -1/2 ) )    
                        .add(
                        bayesFactor
                        );

        const linearFn = weights.transpose().matMul(dataX).add(bias);

        // feeding our linear function to logistic sigmoid function
        const classConditionalProb = this.logisticFn(linearFn);


        // TODO: converting class conditional probabilities into classes 
        

        if (probOrClass === 0)
            return classConditionalProb
    

    }
}