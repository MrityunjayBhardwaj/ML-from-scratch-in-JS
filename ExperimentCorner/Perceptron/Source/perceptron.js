
function perceptron(data){

    const model = {
        weights : [],
        params : {
        }
    }

    this.getWeights = ()=>{return model.weights;}

    this.classify = function(x, weights){
        // return predY {-1,+1}

        const f = tf.matMul(x,weights);

        // if  f_i < 0 == -1 else +1
        const  g  = tf.mul(f, 100).clipByValue(-1, 1);

        return g;
    },
    this.cfn = (y, yPred) =>{

        // finding the nature of all the data points
        const classifiedPts    = tf.pow( tf.add(y, yPred), 2 ).clipByValue(0,1);
        const missClassifiedPts = tf.sub(tf.scalar(1), classifiedPts);

        // add panenty only to the miss classified points:-
        const error = tf.neg( tf.sum( tf.mul( yPred, tf.mul(y, missClassifiedPts) ) ) );

        return error;
    },
    this.updateRule = (x, y, yPred, loss ) =>{
        return tf.sum(tf.mul(x,y), axis=0);
    }

    this.train = (data) => {
        // algorithm:-

        // convert one hot encoded to +1 and -1 encoding scheme
        const modDataY =  data.y.matMul( tf.tensor([[-1],[1]])); // do +1 and -1 encoding

        // calculate weights using gradient descent 
        const calcWeights = optimize( x=data.x, y=modDataY, params={ yPredFn:this.classify, costFn:this.cfn, costFnDx:this.updateRule, learningRate:-1,verbose: true,batchSize:1,epoch:100, threshold: -1} );

        // assigning values to our model for further use.
        model.weights = calcWeights;
        
        return this;
    }

    this. test = (testDataX) => {

        return this.classify(testDataX,model.getWeights());

    }
}