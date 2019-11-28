

function NeuralNetworks(structure = [3, 3, 3], weights /* user can insert weights of pre-trained model */){
    const model = {
        params : {

        },
        weights: [],
        biases: [],

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

        console.log("initial Weights and biases:-");

        for(let i=1;i< structure.length;i++){

            const currNeuronValue = tf.zeros([1, structure[i]]);

            model.neuralNetwork[i-1] = {prepro : currNeuronValue,
                                      output : currNeuronValue};

        //     // initialize weights
        //     // TODO: use a better initialization criterion like Xavier / He initialization.

            model.weights[i-1] = tf.randomNormal([structure[i-1], structure[i]]).mul(2).sub(1);
            model.biases[i-1]  = tf.randomNormal([1, 1]);

            model.weights[i-1].print();
            model.biases[i-1].print();

        }

    }


    this.getWeights   = function(){return model.weights; },
    this.getBiases    = function(){return model.biases; },
    this.getNeuralNet = function(){return model.neuralNetwork;}
    this.getNetworkDx = function(){return model.networkDx;}
    
    this.costFn = function(predY, trueY, params /* any userdefined params for our error function */) {

        return tf.sum( tf.pow(trueY.sub(predY), 2), axis=0 );
    },

    this.costFnDx = function(predY, trueY, params){

        return  predY.sub(trueY) ;
    },

    this.activationFn = function(prepro,output, params /* any userdefined params for this activation function */){

        // using in-built sigmoid function:
        return tf.sigmoid(prepro)

        // using relu
        // return tf.clipByValue(output, 0);
    },

    this.activationFnDx = function(prepro, params){
        // NOTE: prepro is going to be of size N x 1 where N = no. of points M = num of features
        // so please make sure to have an output in dim Nx1 

        return this.activationFn(prepro).mul( tf.tensor(1).sub(this.activationFn(prepro)));

        // relu Dx
        // return prepro > 0
    },

    



    this.train = function(data, params= {learningRate: 0.001, epoch: 100, threshold: 0.01, batchSize: 1.0, verbose: false}){

        // initialize neural network
        const struct = [data.x.shape[1]].concat(structure).concat([data.y.shape[1]]);
        init(struct); 

        // training
        let epoch = nEpoch || 100 ;

        for(let i=0;i< epoch;i++){


            // const shuffledData = tf.shuffle( data);

            // const cBatchData = trainTestSplit(data.x, data.y, .3)[0];
            const cBatchData = data;

            // predict y using the current weights
            let cPredY = this.forwardPass(cBatchData);


            // for(let i=0;i< model.neuralNetwork.length;i++) {

            //     // filtering neuralNetworks
            //     model.neuralNetwork[i].prepro = tf.tensor(model.neuralNetwork[i].prepro.arraySync().map( (cRow, rowIndex)=> { return cRow.map( (cVal, cellIndex) => { return (isNaN(cVal))? 0 : cVal} ) } ));
            //     model.neuralNetwork[i].output = tf.tensor(model.neuralNetwork[i].output.arraySync().map( (cRow, rowIndex)=> { return cRow.map( (cVal, cellIndex) => { return (isNaN(cVal))? 0 : cVal} ) } ));

            // }

            // predYOneHot = tf.tensor(cPredY.arraySync().map( (cRow, rowIndex)=> { return cRow.map( (cVal, cellIndex) => { return (cVal > 0.5)? 1 : 0} ) } ));

            // NOTE: ITS CUSTOM
            //  TODO: generalize this bit:

            // calculate the error:-
            let cLoss = this.costFn(cPredY, cBatchData.y);

            // printing entire report

            if ( ( (i % 10) === 0 ) && verbose )
                console.log("Loss: "+tf.sum(cLoss, axis=0)+" epoch: "+i)


            // calculate the derivatives of loss w.r.t all the neurons.
            // and store them to our model.networkDx array.
            this.backprop(cBatchData, cPredY, params);

        }
    },

    this.test = function(data){

        const dataX = data.x;

        const weights = this.getWeights();
        const biases  = this.getBiases();

        // if (weights.shape[0] === 0){
        //     throw new Error('No weights specified, either specify the weights or train the model');
        // }


        const nLayers = model.neuralNetwork.length; // no of layers:- hidden + 1+outputlayer

        let preLayer = dataX; // here, we are assuming input as a layer but it isn't appear in our model.neuralnetwork

        let predY = 0; 

        for(let i=0; i<nLayers; i++){

            const layerWeights = weights[i];

            // if ( model.biases[i].shape[0] !== dataX.shape[0] )
            const layerBiases  = biases[i];

            const layerPrepro = preLayer.matMul(layerWeights).add(layerBiases);

            // TODO: Add functionality to have different activation function. for each neuron

            const cNeurons = model.neuralNetwork[i].prepro.shape[1];

            let layerOutput =(cNeurons > 1)? this.activationFn( layerPrepro.slice([0, 0], [-1, 1])) : layerPrepro;

            // calculate the activation function for each neuron on layer 'i' sepratly.
            for(let j=1;j< cNeurons;j++){
                const cPrepro = layerPrepro.slice([0,j], [-1,1]);

                if (nLayers-1 === i){
                    layerOutput = layerOutput.concat((cPrepro), axis=1);
                }
                else{
                    layerOutput = layerOutput.concat(this.activationFn(cPrepro), axis=1);

                }

            }

            // TODO: Fix this for other use cases
            // model.neuralNetwork[i].prepro = layerPrepro;
            // model.neuralNetwork[i].output = layerOutput; 

            if(i === nLayers-1)
                predY = layerOutput;

            preLayer = layerOutput;
        }

        return predY;
        

    },

    this.forwardPass = function(data){

        const dataX = data.x;

        const weights = this.getWeights();
        const biases  = this.getBiases();

        // if (weights.shape[0] === 0){
        //     throw new Error('No weights specified, either specify the weights or train the model');
        // }


        const nLayers = model.neuralNetwork.length; // no of layers:- hidden + 1+outputlayer

        let preLayer = dataX; // here, we are assuming input as a layer but it isn't appear in our model.neuralnetwork

        let predY = 0; 

        for(let i=0; i<nLayers; i++){

            const layerWeights = weights[i];

            // if ( model.biases[i].shape[0] !== dataX.shape[0] )
            const layerBiases  = biases[i];

            const layerPrepro = preLayer.matMul(layerWeights).add(layerBiases);

            const cNeurons = model.neuralNetwork[i].prepro.shape[1];

            let layerOutput =(cNeurons > 1)? this.activationFn( layerPrepro.slice([0, 0], [-1, 1])) : layerPrepro;

            // calculate the activation function for each neuron on layer 'i' sepratly.
            for(let j=1;j< cNeurons;j++){
                const cPrepro = layerPrepro.slice([0,j], [-1,1]);

                if (nLayers-1 === i){
                    layerOutput = layerOutput.concat((cPrepro), axis=1);
                }
                else{
                    layerOutput = layerOutput.concat(this.activationFn(cPrepro), axis=1);

                }

            }

            model.neuralNetwork[i].prepro = layerPrepro;
            model.neuralNetwork[i].output = layerOutput; 

            if(i === nLayers-1)
                predY = layerOutput;

            preLayer = layerOutput;
        }

        return predY;
        
    }

    this.backprop = function(data, predY){

        /**
         * TODO: 
         * 1. Calculate the error function dx.
         * 2. recursively calculate the derivative for all the weights for each layer
         * 
         */

        const nLayers = model.neuralNetwork.length; // no of layers:- hidden + 1+outputlayer

        const learningRate = .2/data.x.shape[0];
        // const learningRate = .01;
        // calculating the error function dx;

        const newWeights = [];
        const newBiases  = [];
        
        let preLayerOut = (nLayers-1 > 0)? model.neuralNetwork[nLayers-2].output : data.x;

        // currently using same activation Function. doing some bug fixes
        let lastDx = this.costFnDx(model.neuralNetwork[nLayers-1].output, data.y);
        // NOTE: removed this from lastDx ==> .mul( this.activationFnDx( model.neuralNetwork[nLayers - 1].prepro ) )


        // lastDx = tf.tensor(lastDx.flatten().arraySync().map( function(a,i){return (isNaN(a))? 0 : a})).expandDims(1);

        lastDx = tf.tensor(lastDx.arraySync().map( (cRow, rowIndex)=> { return cRow.map( (cVal, cellIndex) => { return (isNaN(cVal))? 0 : cVal} ) } ) );
        model.networkDx[nLayers -1] = lastDx;

        // updating weights
        newWeights[nLayers-1] = tf.sub( model.weights[nLayers-1], tf.sum(preLayerOut.transpose().matMul(lastDx)) .mul(learningRate));
        newBiases[nLayers-1]  = tf.sub( model.biases[nLayers -1], tf.sum(lastDx, axis=0).mul(learningRate));


        for(let l=nLayers-2; l>=0; l--){

            let currLayerDx = model.networkDx[l+1]
            .matMul( model.weights[l+1].transpose() )
            .mul( this.activationFnDx( model.neuralNetwork[l].prepro ) );

            // inserting this derivative to our networkDerivative array for future use.

            currLayerDx = tf.tensor(currLayerDx.arraySync().map( (cRow, rowIndex)=> { return cRow.map( (cVal, cellIndex) => { return (isNaN(cVal))? 0 : cVal} ) } ));

            model.networkDx[l] = currLayerDx;

            // updating weights

            preLayerOut = (l > 0)? model.neuralNetwork[l-1].output : data.x;

            newWeights[l] = tf.sub( model.weights[l], tf.sum(preLayerOut.transpose().matMul( currLayerDx ), axis=0).mul(learningRate));
            newBiases[l]  = tf.sub( model.biases[l],  tf.sum((currLayerDx), axis=0).mul(learningRate));

        }


        // making new Weights into current weights;
        model.weights = newWeights;
        model.biases  = newBiases;

    }
}