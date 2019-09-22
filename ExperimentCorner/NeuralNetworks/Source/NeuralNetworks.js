

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

    
        this.backprop(  );


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

        let output = 0;
        for(let i=1; i<nLayers; i++){

            const cWeights = weights.slice([0,0], [-1, i]);

            const prepro = preLayer.matMul(cWeights);
            // TODO: Add functionality to have different activation function. for each neuron
            const activation = this.activationFn(prepro);

            if(i++ === nLayers)
                output = activation;
        }

        // calculate the error:-
        this.lossFn(output, dataY)

        
    }
}