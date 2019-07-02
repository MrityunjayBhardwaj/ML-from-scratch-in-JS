
/**
 * 
 * @param {string} type type of const function we require
 * @summary given the type this function returns a cost function which takes 3 params
 * data 'y' values and predicted 'y' value.
 */
function costFn(type){
    if (type === "mse"){

        // mean-squared-error (yPred - y)**2
    return function(y,yPred){
        return  tf.sum( tf.pow( tf.sub(yPred , y ) , 2 ) )
    };
    
    }

    // add other cost function like R^2 etc.

}

/**
 * 
 * @param {string} type type of const function we require
 * @summary given the type this function returns a cost function derivatives which takes 3 params
 * data x and y values and predicted 'y' value.
 */
function costFnDerivatives(type){
    if(type === "mse"){
        return function(x,y,yPred) {

            // d/dx of MSE w.r.t 'w' : (yPred - y)x
            return tf.matMul( x.transpose() , tf.sub( yPred , y ) );
            
        }
    }
}

/**
 * 
 * @param {object} x tf.tensor input data X
 * @param {object} y input data Y of type tf.tensor 
 * @param {object} params important paramters for our optimizer
 * @param {function } params.yPridFn this function is provided with dataX and weights and returns the prediction Y
 * @param {function } params.costFn function which calculates the cost function given the data Y and predicted Y
 * @param {function}  params.costFnDx function which spits out the derivative of the cost function 
 * @param {function}  params.callback a simple callback function which gets called at every epoch
 * @param {null}      params.yPred helpful for transfer learning
 * @param {object}    params.weights if you have pretrained weights vector, you can plug it in here, useful for transfer learning 
 * @param {number}    params.epoch maximum number of gradient descent steps
 * @param {number}    params.learningRate 
 * @param {number}    params.threshold the maximum amount allowable difference between prediction and truth. 
 * @summary this function tries to find the weights which maximize/minimize the const function and using the parameters
 */
function optimize(x,y,
                  params={
                            yPridFn  = function(x,w){
                                     return tf.matMul(w,x) 
                                    },
                            costFn   = costFn('mse'),
                            costFnDx = costFnDerivatives('mse'),
                            callback = null,
                            yPred    = null,
                            weights  = null,
                            epoch    = 1000,
                            learningRate = 0.001,
                            threshold    = 1e-3
                        }
                ) {


    // initializing weights vector tf.matMul( x, oldWeights).
    if (!weights){
        // works only if x.shape = [m,n...] where, m == no. of training samples.
        weights =  tf.randomNormal([x.shape[1],1]);
    }

    let oldWeights = weights;
    let oldBias    = tf.randomUniform([1,1]);
    for(let i = 0;i<epoch;i++){

        // calculating new prediction and loss function.
        yPred = yPridFn(x,oldWeights);
        const Loss = costFn(y,yPred);

        // checking convergence.
        if (Loss.arraySync() < threshold){
            yPred.print();
            return oldWeights;
        }

        // Calculating and Updating new Weights
        const weightDx = costFnDx(x,y,yPred);
        const newWeights = tf.sub( oldWeights , tf.mul( tf.scalar( learningRate ) , weightDx ) );
        
        // reAssigning weights 
        oldWeights = newWeights; 
        
        // invoke the callback function 
        if (callback !== null)
        callback(x,y,yPred,oldWeights);
    }

    return [oldWeights,oldBias];
}

