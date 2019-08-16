
function LogisticRegression(){
    const model = {
        weights: [],
        params : {
            learningRate : 0,
            batchSize : 100,
            epoach : 100,
            threshold: .01,
        },
    }

    this.getWeights = function(){
        return model.weights;
    }
    this.logisticFn = (threshold=0.5,convert2Class=1, permWeights=null) => {

            return function(x,weights=permWeights){

            // calculating logistic function
            const logOdds = tf.matMul( x, weights ); // here we are using w^Tx in-place of log odds
            const expLogOdds = tf.exp( tf.neg( logOdds ) );
            let logit = tf.div( 1, tf.add( 1, expLogOdds ) );

            // converting probability into prediction (Classes)
            if(convert2Class)
            {
                const thCenter  = tf.sub( logit,( threshold ) );
                const predClass = tf.pow( tf.clipByValue(tf.mul(thCenter, 10000000 ), 0, 1 ), 1 );

                return predClass
            }

            return logit;
        }
    },
    this.lfn = (x,weights=model.weights) => {

            // calculating logistic function
            const logOdds = tf.matMul( x, weights ); // here we are using w^Tx in-place of log odds
            const expLogOdds = tf.exp( tf.neg( logOdds ) );
            let logit = tf.div( 1, tf.add( 1, expLogOdds ) );

            return logit;
        }

    this.train = function(data,params ={} ){

        // augmenting params object to insert important parameters that are important for our model training.
        params.threshold = params.threshold || model.params.threshold;
        params.yPredFn = this.lfn;

        // convert data.y one hot into binary and calaculate weights
        const dataBinaryY    = oneHot2Class( data.y );
        const trainedWeights = optimize( data.x, dataBinaryY, params );

        // calcWeights.print()
        model.weights = trainedWeights;
        return this;
    }

    // { weights:null, threshold:null, probOrClass:1}
    this.classify = function(testDataX, params={}){
        const { 
                weights     = model.weights, 
                threshold   = model.params.threshold,
                probOrClass = 1 
                } = params;

        const prob = this.lfn( testDataX, weights );
        // converting probability into prediction (Classes)
        if(probOrClass)
        {
            const thCenter  = tf.sub( prob,( threshold ) );
            const predClass = tf.pow( tf.clipByValue(tf.mul(thCenter, 10000000 ), 0, 1 ), 1 );

            const oneHot0 = tf.ones(predClass.shape).sub( predClass);
            const oneHotPredClass = oneHot0.concat( predClass, axis=1 );

            return oneHotPredClass; 
        }

        return prob;
    }
}



