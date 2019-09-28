

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
    
    this.costFn = function(predY, trueY, params /* any userdefined params for our error function */) {

        return trueY.sub(predY).mul(2).pow(2);
    },

    this.costFnDx = function(predY, trueY, params){

        return trueY.sub(predY);
    },

    this.activationFn = function(prepro, params /* any userdefined params for this activation function */){

        // using sigmoid
        return tf.exp(tf.neg(prepro)).add(1).pow(-1);
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

        const learningRate = 0.0001;
        // calculating the error function dx;

        // currently using same activation Function. 
        let lastDx = this.costFnDx(predY, data.y).mul( this.activationFnDx( model.neuralNetwork[nLayers - 1].prepro ) );

        model.networkDx[nLayers -1] = lastDx;

        // updating weights
        model.weights[nLayers-1] = tf.sub( model.weights[nLayers-1], model.neuralNetwork[nLayers-2].output.transpose().matMul(lastDx));

        for(let l=nLayers-2; l>=0; l--){

            const currLayerDx = model.networkDx[l+1]
            .matMul( model.weights[l+1].transpose() )
            .mul( this.activationFnDx( model.neuralNetwork[l].prepro ) );

            // inserting this derivative to our networkDerivative array for future use.
            model.networkDx[l] = currLayerDx;

            // updating weights
            if (l > 0)
                model.weights[l] = tf.sub( model.weights[l], model.neuralNetwork[l-1].output.transpose().matMul( currLayerDx).mul(learningRate));
            else{
                model.weights[l] = tf.sub( model.weights[l], data.x.transpose().matMul( currLayerDx).mul(learningRate));

            }

        }

    }


    this.train = function(data){

        // initialization
        const struct = [data.x.shape[1]].concat(structure).concat([data.y.shape[1]]);
        init(struct); 

        // training
        let epoch = 10;
        let weights = 0;

        for(let i=0;i< epoch;i++){

            // predict y using the current weights
            const cPredY = this.forwardPass(data);

            // NOTE: ITS CUSTOM
            //  TODO: generalize this bit:
            const predYOneHot = pred2Class(cPredY.slice([0, 0], [-1, 1]), threshold=0.5, oneHot=false)
                                .concat( pred2Class(cPredY.slice([0, 1], [-1,1]), threshold=0.5, oneHot=false), axis=1);

            // calculate the error:-
            const cLoss = this.costFn(predYOneHot, data.y);

            console.log("Loss: "+tf.sum(cLoss)+" epoch: "+i)

            // calculate the derivatives of loss w.r.t all the neurons.
            // and store them to our model.networkDx array.
            this.backprop(data, predYOneHot);

            // TODO: use the networkDx to update the weights.



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