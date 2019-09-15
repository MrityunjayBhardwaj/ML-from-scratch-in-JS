

function NeuralNetworks(){
    this.model = {
        params : {

        },
        weights: tf.tensor(),

    }

    this.getWeights = function(){return model.weights}

    this.derivatives = function() {

    },

    this.costFn = function() {

    },

    this.train = function(data){
        


    }

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