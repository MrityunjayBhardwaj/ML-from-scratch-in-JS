/**
 * lr_pred = LR(tX,tY,{epoch,optimizer,callback}).train().test(testX);
 * 
 */

// Only for Learning Purposes
/**
 * 
 * @param {object} x features/independent_var for training
 * @param {object} y output/dependent_var for training. 
 * @param {object} params {epoch : number of epoch , optimizer : 'sgd' , callback : function to call after each epoach.}
 */
function LinearRegression(){
    const model = {
        weights : []
    }

    /**
     * @param {object} data x and y data values,formatted like this:- {x: dataX, y : dataY}
     * @param {object} params a collection of parameters useful in optimizing our objective function
     * @summary this funciton takes 2 arguments and spits out the best possible wights vector , first argument is just an data Inut matrix 
     * the second argument is a object which conating some important parameters, they are:-
     * 'epoch' = number of iterations;
     * 'optimizer' = type of optimizer for our gradient descent; 
     * 'threshold' = maximum allowable loss b/w y and predicted y;
     * 'learningRate' = self explanatory;
     */
    this.train =  function( data, params = {} ){
        
        const {epoch = 100,optimizer = "sgd",threshold = 1e-3 ,learningRate = 1e-4} = params;

        const dataXWithBias = matrixX.concat( tf.ones([matrixX.shape[0],1]), axis=1);

        let calcWeights = optimize(dataXWithBias,data.y,costFn("mse"),costFnDerivatives("mse"),null,null,null,5000,0.003);

        console.log("weights: ");
        calcWeights.print();

        model.weights = calcWeights;

        const k =  this.test(dataXWithBias);
        const sol  = tf.sub(1, tf.sum(tf.pow( tf.sub( data.y, k ), 2 )));
        sol.print();
    }

    this.test = function(testDataX){
        return tf.matMul(testDataX, model.weights);
    }

}
