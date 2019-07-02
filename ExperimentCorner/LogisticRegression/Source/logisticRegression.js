
function LogisticRegression(){
    const model = {
        weights: [],
        params : {
            learningRate : 0,
            batchSize : 100,
            epoach : 100,
        },
    }

    this.logisticFn = function(x,weights,threshold=.5){

        // calculating logistic function
        const logOdds = tf.matMul(weights,x);
        const expLogOdds = tf.exp( tf.neg( logOdds ) );
        const logit = tf.div( expLogOdds, tf.sub( 1, expLogOdds ) );

        // converting probability into prediction
        let yPred = tf.tensor( logit.arraySync().map( (prob) => (prob > threshold)*1 ) );
        return yPred;
    }

    this.train = function(data){
        
        // convert data.y one hot into binary and calaculate weights
        const dataBinaryY = tf.tensor( data.y.arraySync().map( (cVec) => { return cVec.indexOf(1) } ) );
        const calcWeights = optimize( data.x, dataBinaryY, { yPredFn= this.logisticFn } );

        model.weights = calcWeights;
        return this;
    }

    this.classify = function(testDataX){
        return this.logisticFn(testDataX, model.weights);
    }
}
