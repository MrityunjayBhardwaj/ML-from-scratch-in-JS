

function NeuralNetworks(structure = [3, 3, 3], weights /* user can insert weights of pre-trained model */){
    const model = {
        params : {

        },
        weights: tf.tensor([]),

        neuralNetwork : [],

        /**
         * Structure:-
         * - 3 x 3 neuron
         * 
         *  [. . . ]
         *  [. . . ]
         *  [. . . ]
         * 
         * where, each [.] => {prepro, output, weights} and for the first column the weights are set to 'None'
         * because there are 2 weight matrix connecting layer 0 to 1 and 1 to 2.
         * 
         */

        networkDx : [] // derivative of all the prepro stage w.r.t loss function for all the layers.

    };

    function init(structure) {
        // initialize the weights.

        for(let i=1;i< structure.length;i++){

            const currNeuronValue = tf.randomNormal([1, structure[i]]);

            model.neuralNetwork[i-1] = {prepro : currNeuronValue,
                                      output : currNeuronValue};

            // initialize weights
            // TODO: use a better initialization criterion like Xavier / He initialization.

            model.weights[i-1] = tf.randomNormal([structure[i-1], structure[i]]);

        }


    }


    this.getWeights = function(){ return model.weights; },
    this.getNeuralNet = function(){return model.neuralNetwork;}
    this.getNetworkDx = function(){return model.networkDx;}
    
    this.costFn = function(predY, trueY, params /* any userdefined params for our error function */) {

        return trueY.sub(predY).mul(2).pow(2);
    },

    this.costFnDx = function(predY, trueY, params){

        return trueY.sub(predY);
    },

    this.activationFn = function(prepro, params /* any userdefined params for this activation function */){

        // using sigmoid
        return tf.tensor(1).sub(tf.exp(tf.neg(prepro))).pow(-1);
    },

    this.activationFnDx = function(prepro, params){
        // NOTE: prepro is going to be of size N x 1 where N = no. of points M = num of features
        // so please make sure to have an output in dim Nx1 

        return tf.exp(tf.neg(prepro)).mul( tf.tensor(1).sub(tf.exp(tf.neg(prepro))));
    },

    
    this.backprop = function(data, predY){

        /**
         * TODO: 
         * 1. Calculate the error function dx.
         * 2. recursively calculate the derivative for all the weights for each layer
         * 
         */

        const nLayers = model.neuralNetwork.length; // no of layers:- hidden + 1+outputlayer

        const learningRate = 0.001;
        // calculating the error function dx;

        
        let preLayerOut = (nLayers-2 > 0)? model.neuralNetwork[nLayers-2].output : data.x;

        // currently using same activation Function. 
        let lastDx = this.costFnDx(predY, data.y).mul( this.activationFnDx( model.neuralNetwork[nLayers - 1].prepro ) );

        lastDx = tf.tensor(lastDx.flatten().arraySync().map( function(a,i){return (isNaN(a))? 0 : a})).expandDims(1);

        model.networkDx[nLayers -1] = lastDx;

        // updating weights
        model.weights[nLayers-1] = tf.sub( model.weights[nLayers-1], preLayerOut.transpose().matMul(lastDx).mul(learningRate/data.x.shape[0]));

        for(let l=nLayers-2; l>=0; l--){

            const currLayerDx = model.networkDx[l+1]
            .matMul( model.weights[l+1].transpose() )
            .mul( this.activationFnDx( model.neuralNetwork[l].prepro ) );

            // inserting this derivative to our networkDerivative array for future use.


            model.networkDx[l] = currLayerDx;

            // updating weights

            preLayerOut = (l > 0)? model.neuralNetwork[l-1].output : data.x;

            model.weights[l] = tf.sub( model.weights[l], preLayerOut.transpose().matMul( currLayerDx ).mul(learningRate/data.x.shape[0]));
            // if (l > 0)
            //     model.weights[l] = tf.sub( model.weights[l], model.neuralNetwork[l-1].output.transpose().matMul( currLayerDx ).mul(learningRate/ data.x.shape[0]));
            // else{
            //     model.weights[l] = tf.sub( model.weights[l], data.x.transpose().matMul( currLayerDx ).mul(learningRate/ data.x.shape[0]));

            // }

        }

    }


    this.train = function(data){

        // initialization
        const struct = [data.x.shape[1]].concat(structure).concat([data.y.shape[1]]);
        init(struct); 

        // training
        let epoch = 100;

        for(let i=0;i< epoch;i++){
            // const cBatchData = trainTestSplit(data.x, data.y, .6)[0];
            const cBatchData = data;

            // predict y using the current weights
            const cPredY = this.forwardPass(cBatchData);

            // NOTE: ITS CUSTOM
            //  TODO: generalize this bit:
            // const predYOneHot = pred2Class(cPredY.slice([0, 0], [-1, 1]), threshold=0.5, oneHot=false)
            //                     .concat(pred2Class(cPredY.slice([0, 1], [-1,1]), threshold=0.5, oneHot=false), axis=1);

            const predYOneHot = tf.clipByValue(tf.round(cPredY), 0,1);
            // calculate the error:-
            const cLoss = this.costFn(predYOneHot, cBatchData.y);

            // printing entire report
            console.log("Loss: "+tf.sum(cLoss).div(cBatchData.x.shape[0])+" epoch: "+i)
            // console.log('cBatchData.x: '+cBatchData.x.flatten().arraySync());
            // console.log('model.weights[0]: '+this.getWeights()[0].flatten().arraySync());
            // console.log('model.neuralNetwork[0].prepro: '+this.getNeuralNet()[0].prepro.flatten().arraySync());
            // console.log('model.neuralNetwork[0].output: '+this.getNeuralNet()[0].output.flatten().arraySync());
            // console.log('predYOneHot: ', predYOneHot.flatten().arraySync());
            // console.log('cBatchData.y: '+cBatchData.y.flatten().arraySync());
            

            // calculate the derivatives of loss w.r.t all the neurons.
            // and store them to our model.networkDx array.
            this.backprop(cBatchData, predYOneHot);
            // console.log('model.networkDx[0]: ', this.getNetworkDx()[0].flatten().arraySync())



        }
    },

    this.forwardPass = function(data){

        const dataX = data.x;

        const weights = this.getWeights();

        // if (weights.shape[0] === 0){
        //     throw new Error('No weights specified, either specify the weights or train the model');
        // }


        const nLayers = model.neuralNetwork.length; // no of layers:- hidden + 1+outputlayer

        let preLayer = dataX; // here, we are assuming input as a layer but it isn't appear in our model.neuralnetwork

        let predY = 0; 

        for(let i=0; i<nLayers; i++){

            const layerWeights = weights[i];

            const layerPrepro = preLayer.matMul(layerWeights);

            // TODO: Add functionality to have different activation function. for each neuron

            let layerOutput = this.activationFn( layerPrepro.slice([0, 0], [-1, 1]));

            const cNeurons = model.neuralNetwork[i].prepro.shape[1];

            // calculate the activation function for each neuron on layer 'i' sepratly.
            for(let j=1;j< cNeurons;j++){
                const cPrepro = layerPrepro.slice([0,j], [-1,1]);

                // if (i === nLayers-1){
                //     layerOutput = layerOutput.concat(cPrepro, axis=1);
                //     continue;
                // }

                layerOutput = layerOutput.concat(this.activationFn(cPrepro), axis=1);
            }

            model.neuralNetwork[i].prepro = layerPrepro;
            model.neuralNetwork[i].output = layerOutput; 

            if(i === nLayers-1)
                predY = layerOutput;

                preLayer = layerOutput;
        }

        return predY;
        
    }
}