

function NeuralNetworks(){
    this.model = {
        params : {

        },
        weights: tf.tensor(),

        nerualNetwork : {},

        derivativeNetwork : {}

    },

    

    this.getWeights = function(){ return model.weights; }

    this.derivatives = function() {

    },

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

        let lastDx = this.costFnDx(predY).mul( this.activationFnDx( this.model.nerualNetwork[nLayers - 1].prepro ) );

        for(let l=nLayers-1;l>=0;l--){

            const currLayerDx = this.model.nerualNetwork[l+1].weights.matMul(lastDx).mul( this.activationFnDx( this.model.nerualNetwork[l].prepro ) )

            lastDx = currLayerDx;

        }

    }


    this.train = function(data){
        // TODO: do backpropagation.

        


    },

    this.forwardPass = function(data, weights = null){

        const dataX = data.x;
        const dataY = data.y;

        const weights = weights || this.getWeights();

        if (weights.shape[0] === 0){
            throw new Error('No weights specified, either specify the weights or train the model');
        }

        // if we have weights :-

        const nLayers = model.params.networkStruct.length; // no of layers:- hidden + 1+outputlayer

        let preLayer = dataX;

        let output = 0;
        for(let i=1; i<nLayers; i++){

            const cWeights = weights.slice([0,0], [-1, i]);

            const prepro = preLayer.matMul( cWeights );
            // TODO: Add functionality to have different activation function. for each neuron
            const activation = this.activationFn(prepro);

            if(i++ === nLayers)
                output = activation;
        }

        // calculate the error:-
        this.lossFn(output, dataY)

        
    }
}