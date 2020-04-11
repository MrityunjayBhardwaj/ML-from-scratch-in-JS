const MultivariateNormal = window.MultivariateNormal.default;

function GMM(inputMean= [], inputCovariance= [], inputMixingCoeff= []){

    this.model= {

        mean : inputMean,
        covariance: inputCovariance,
        mixingCoeff: inputMixingCoeff,
    },


    this.train = function(tensor,/* no. of components*/k=5,epsilon=0.1, maxItrs=10 ){

        // initialize mean, covariance and mixing coefficients

        let N = tensor.shape[0]; // number of datapoints
        let mean = new Array(k);
        let covariance = new Array(k);
        let mixingCoeff = new Array(k);
        let responsiblities = new Array(k);

        let prevLoglikelihood = 0;

        let currTotalMix = 0;


        // if the model parameters are specified then use them
        if (this.model.mean !== [] && false){
            // mean = this.model.mean;
            // covariance = this.model.covariance;
            // mixingCoeff = this.model.mixingCoeff;
        }
        else{
            // otherwise, generate them randomly at first

            for(let i=0;i< k;i++){

                // generating random mean
                mean[i] = tf.randomNormal([1, tensor.shape[1]]);
                // mean[i] = tf.zeros([1, tensor.shape[1]]);

                // generate random covariance matrix  A.T * A where, A = n by n matrix,  gives a positive semidefinite matrix
                // covariance[i] = tf.randomNormal([tensor.shape[1], tensor.shape[1]]);
                // covariance[i] = covariance[i].transpose().matMul(covariance[i]);
                covariance[i] = tf.eye(tensor.shape[1])

                // generating random mixing coefficients and making sure that all sum up to 1
                if (i === (k-1)){
                    mixingCoeff[i] = tf.tensor([1-currTotalMix]);
                    mixingCoeff[i].print();
                    continue;
                }
                mixingCoeff[i] = tf.randomUniform([1],0,(1-currTotalMix));
                currTotalMix  += mixingCoeff[i].flatten().arraySync()[0];

                mixingCoeff[i].print();

            }
        }

        let itr = 0;
        while(itr < maxItrs){

            // E-step: Evaluate the responsibilities using the current parameter values
            let normalizationConstant = tf.zeros([N,1]); // sum of all the responsibilities

            for(let i=0;i< k;i++){

                const normalDist = new multivariateGaussian(mean[i], covariance[i]);

                responsiblities[i] = normalDist.getProbability(tensor).mul(mixingCoeff[i]);
                responsiblities[i] = responsiblities[i].reshape([responsiblities[i].shape[0], 1]);

                normalizationConstant = normalizationConstant.add(responsiblities[i]);

            }

            // normalizing the responsibilities
            for(let i=0;i<k;i++){
                responsiblities[i] = responsiblities[i].div(normalizationConstant);
            }

            // M-step: Re-estimate the parameters using the current responsibilities
            for(let i=0;i<k;i++){

                // update mean
                const currCompSize =  responsiblities[i].sum(axis=0);// no. of points belongs to the current component
                mean[i] = tensor.mul(responsiblities[i]).sum(axis=0).div(currCompSize);

                // update covariance
                const meanCenteredData = tensor.sub(mean[i]);
                covariance[i] = meanCenteredData.mul(responsiblities[i]).transpose().matMul(meanCenteredData).sum(axis=1).div(currCompSize);

                covariance[i] = meanCenteredData.expandDims().reshape([meanCenteredData.shape[0], meanCenteredData.shape[1], 1])
                                .matMul(
                                        meanCenteredData.expandDims().reshape([meanCenteredData.shape[0], 1, meanCenteredData.shape[1]])
                                    )
                                .mul(
                                    responsiblities[i].expandDims().reshape([responsiblities[i].shape[0], 1, responsiblities[i].shape[1]])
                                    )
                                .sum(axis=0)
                                .div(currCompSize);
                // update responsibilites
                mixingCoeff[i] = currCompSize.div(N);

            }

            // TODO: remove these guys..
            myMeans.push(mean.slice());
            myCovariance.push(covariance.slice());
            myMixingCoeff.push(mixingCoeff.slice());

            // Evaluate the log likelihood

            let sumOfAllProb = tf.zeros([tensor.shape[0],1]);

            for(let i=0;i<k;i++){

                const normalDist = new multivariateGaussian(mean[i], covariance[i]);
                 sumOfAllProb = sumOfAllProb.add( normalDist.getProbability(tensor).expandDims().transpose().mul(responsiblities[i]) );
            }
            const currLoglikelihood = tf.log(sumOfAllProb).sum(axis=0).flatten().arraySync()[0];

            // TODO: work on the convergence criterion

            console.log(itr+ ") Diff: ",Math.abs(Math.abs(prevLoglikelihood) - Math.abs(currLoglikelihood)), 'currLoglikelihood: '+currLoglikelihood);

            if (isNaN(mean[mean.length-1].flatten().arraySync()[0])){

                const lastItrParams = {mean: myMeans[myMeans.length-3], covariance: myCovariance[myMeans.length-3], mixingCoeff: myMixingCoeff[myMeans.length-3]}

                this.model.mean = lastItrParams.mean;
                this.model.covariance  = lastItrParams.covariance;
                this.model.mixingCoeff = lastItrParams.mixingCoeff;

                return lastItrParams;

            }
            if ( Math.abs(Math.abs(prevLoglikelihood) - Math.abs(currLoglikelihood)) < epsilon)
                break;
            
            prevLoglikelihood = currLoglikelihood

            itr++;

        }

        this.model.mean = mean;
        this.model.covariance  = covariance;
        this.model.mixingCoeff = mixingCoeff;


        return { mean: mean, covariance: covariance, mixingCoeff: mixingCoeff};
    },
    this.test = function(tensor){

        let dataProbability = tf.zeros([tensor.shape[0],1]);

        const k = this.model.mean.length;
        for(let i=0;i<k;i++){

            const currGaussian = new multivariateGaussian(this.model.mean[i], this.model.covariance[i]);

            const weightedProb = currGaussian.getProbability(tensor).expandDims().transpose().mul(this.model.mixingCoeff[i]);
            dataProbability = dataProbability.add(weightedProb);
        }

        return dataProbability;

    }

    this.generateSamples = function(noOfSamples){
        // TODO: Complete this using ancestral sampling...
    }

    this.exportModelParams = function(){
        // convert and save model
        let myConvertedMeans = [];
        let myConvertedCovariance = [];
        let myConvertedMixingCoeff = [];
        for(let i=0;i<myMeans.length;i++){
        
            const currItrMeans = [];
            const currItrCovariance = [];
            const currItrMixingCoeff = [];
            
            for(let j = 0;j< myMeans[0].length;j++){
                currItrMeans.push( myMeans[i][j].arraySync());
                currItrCovariance.push( myCovariance[i][j].arraySync());
                currItrMixingCoeff.push( myMixingCoeff[i][j].flatten().arraySync()[0]);
            }
            
            myConvertedMeans.push(currItrMeans);
            myConvertedCovariance.push(currItrCovariance);
            myConvertedMixingCoeff.push(currItrMixingCoeff);
        }

        
        return {mean: myConvertedMeans, covariance: myConvertedCovariance, mixingCoeff: myConvertedMixingCoeff};

    }

}