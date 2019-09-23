

function NeuralNetworks(structure = [3, 3, 3], weights /* user can insert weights of pre-trained model */){
    this.model = {
        params : {

        },
        weights: tf.tensor(),

        nerualNetwork : {},

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

    },

    function init() {
        // initialize the weights.

        // initializing model.neuralNewtwork. each 
        for(let i=0;i< structure.length;i++){

            const currNeuronValue = tf.zeros([structure[i], 1]);

            this.model.neuralNetwork[i].prepro = currNeuronValue;
            this.model.neuralNetwork[i].output = currNeuronValue;

            if (i >= 1 ){
                // initialize weights
                // TODO: use a better initialization criterion like Xavier / He initialization.

                this.model.neuralNetwork[i].weights = tf.zeros([structure[i-1], structure[i]]);
            }

        }


    }

    this.getWeights = function(){ return model.weights; }


    this.costFn = function(predY, trueY, params /* any userdefined params for our error function */) {

    },
    this.costFnDx = function(predY, trueY, params){

    },

    this.activationFn = function(prepro, params /* any userdefined params for this activation function */){

    },

    this.activationFnDx = function(prepro, params){
        // NOTE: prepro is going to be of size N x 1 where N = no. of points M = num of features
        // so please make sure to have an output in dim Nx1 

    },




    this.backprop = function(predY){

        /**
         * TODO: 
         * 1. Calculate the error function dx.
         * 2. recursively calculate the derivative for all the weights for each layer
         * 
         */

        const nLayers = 10;

        // calculating the error function dx;

        // currently using same activation Function. 
        let lastDx = this.costFnDx(predY).mul( this.activationFnDx( this.model.nerualNetwork[nLayers - 1].prepro ) );

        this.model.networkDx[nLayers -1] = lastDx;

        for(let l=nLayers-2; l>=0; l--){

            const currLayerDx = this.model.nerualNetwork[l+1].weights.matMul( this.model.networkDx[l+1] ).mul( this.activationFnDx( this.model.nerualNetwork[l].prepro ) )

            // inserting this derivative to our networkDerivative array for future use.
            this.model.networkDx[l] = currLayerDx;
        }

    }


    this.train = function(data){
        // TODO: do backpropagation.

        let epoach = 10;

        let weights = 0;
        for(let i=0;i< epoach;i++){

            // predict y using the current weights
            const cPredY = this.forwardPass(data, weights);

            // calculate the error:-
            const cLoss = this.lossFn(cPredY, dataY);

            // calculate the derivatives of loss w.r.t all the neurons.
            // and store them to our this.model.networkDx array.
            this.backprop(cPredY);


        }
    },

    this.forwardPass = function(data, weights = null){

        const dataX = data.x;
        const dataY = data.y;

        const weights = weights || this.getWeights();

        // if (weights.shape[0] === 0){
        //     throw new Error('No weights specified, either specify the weights or train the model');
        // }


        const nLayers = model.neuralNetwork.length; // no of layers:- hidden + 1+outputlayer

        let preLayer = dataX; // here, we are assuming input as a layer but it isn't appear in our model.neuralnetwork

        let predY = 0; 

        for(let i=1; i<nLayers; i++){

            const layerWeights = weights[i];

            const layerPrepro = preLayer.matMul(layerWeights);
            // TODO: Add functionality to have different activation function. for each neuron

            let layerOutput = tf.tensor();

            // calculate the activation function for each neuron on layer 'i' sepratly.
            for(let j=0;j< nLayers[i];j++){
                const cPrepro = layerPrepro.slice([0,j], [-1,1]);
                layerOutput = layerOutput.concat(this.activationFn(cPrepro));
            }

            if(i++ === nLayers)
                predY = layerOutput;
        }

        return predY;
        
    }
}