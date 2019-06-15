
    function costFn(type){
        if (type === "mse"){

            // mean-squared-error (yPred - y)**2
        return function(y,yPred){
          return  tf.sum( tf.pow( tf.sub(yPred , y ) , 2 ) )
        };
        
        }

        // add other cost function like R^2 etc.

    }

    function costFnDerivatives(type){
        if(type === "mse"){
            return function(x,y,yPred) {

                // d/dx of MSE : (yPred - y)x
                return tf.matMul( x.transpose() , tf.sub( yPred , y ) );
                
            }
        }
    }
function optimize(x,y,costFn,costFnDx,callback = null,yPred = null ,weights = null,epoch= 1000,learningRate = 1e-4,threshold = 1e-3) {

    
    if (!weights){
        console.log(x.shape,y.shape);

        // works only if x.shape = [m,n...] where, m == no. of training samples.
        x = tf.concat( [ x , tf.ones([x.shape[0],1]) ] , axis = 1);
        weights =  tf.randomUniform([x.shape[1],1]);

        console.log(x.shape,weights.shape)
    }
    let oldWeights = weights;
    for(let i = 0;i<epoch;i++){

        yPred = tf.matMul( x, oldWeights);
        
        // yPred.print();
        const Loss = costFn(y,yPred);

        Loss.print();
        if (Loss.arraySync() < threshold){
            yPred.print();
            return oldWeights;
        }

        // wNew = wOld = alpha*dL/dw
        const weightDerivative = costFnDx(x,y,yPred);
        const newWeights = tf.sub( oldWeights , tf.mul( tf.scalar( learningRate ) , weightDerivative ) );

        oldWeights = newWeights; 
        
        // invoke the callback function 
        if (callback !== null)
        callback(x,y,yPred,oldWeights);
    }

    return oldWeights;
}

