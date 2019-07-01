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

 
function PolynomialRegression(x,y,degree = 3){
    // TODO: check if 'x' and 'y' are tf.tensor objects.


    const trainX = x;
    const trainY = y;

    this.train =  function(params = {}){
        
        const {epoch = 100,optimizer = "sgd",threshold = 1e-3 ,learningRate = 1e-4} = params;

        const polyVec  = tf.linspace(1, degree, degree).expandDims(1); // array of all the powers 02Degree

        polyVec.print();
        // console.log(polyVec.shape)
        x = x.pow(polyVec.transpose());
    
        x.print();
        let calcWeights = optimize(x,y,costFn("mse"),costFnDerivatives("mse"));

        calcWeights.print();

    }

    this.test = function(){

    }

}

// app.js

// regression

