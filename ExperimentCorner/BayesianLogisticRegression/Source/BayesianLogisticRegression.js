function BayesianLogisticRegression(){

    model = {
        predictiveProb : 0,
        parameterPDF: {
            mean : 0,
            covariance: 0
        }
    }

    this.logisticFn = function(a){
        return tf.exp(-a).add(1).pow(-1);
    }

    this.parameterPDF = function(data){

        // hyperparameters for prior distribution.
        const hypParams = {m_0: 0, S_0: 0};


        // TODO: Calculate the W_MAP>>>>

        const currY = 0 // specify the current class of the data.
        const weightMAP = hypParams.m_0.sub( S_0.mul(tf.sum(data.x - currY.mul(data.x))));

        const predY = this.logisticFn( weightMean.matMul(data.x) );
        const weightCovariance = hypParams.S_0.add(
          tf.sum(
            predY.mul(
              tf.tensor(1)
                .sub(predY)
                .matMul(data.x.matMul(data.x.transpose()))
            )
          )
        );

        model.parameterPDF.mean = weightMean;
        model.parameterPDF.covariance = weightCovariance; 


        // TODO: Create a function to generate weights from the parameterPDF.
        // TODO: Create a function to output the probability of of given input weight.

    }
    this.train = function(data){
        const lambda = 1/8; // parameter for inverse probit function.

        const dataX = data.x;

        const meanWeight = model.parameterPDF.mean; // more presicely its MAP weight.
        const weightCovariance = model.parameterPDF.covariance; 

        const mean = meanWeight.matMul( dataX );
        const variance = dataX.matMul(weightCovariance).matMul( dataX );


        const kai = variance.mul(lambda).add(1).pow(-1/2);

        const predictiveProb = this.logisticFn( kai.mul(mean) );

        model.predictiveProb = predictiveProb;
        // return predictiveProb;
    }

}