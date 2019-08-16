
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
    }

    this.train = function(data,callbackFn){
        
        // convert data.y one hot into binary and calaculate weights
        const dataBinaryY = tf.tensor( data.y.arraySync().map( (cVec) => { return cVec.indexOf(1) } ) ).expandDims(1);
        const calcWeights = optimize( data.x, dataBinaryY, { 
            yPredFn: this.logisticFn(model.params.threshold, convert2Class=0), 
            // costFn: this.logisticFn(model.params.threshold, convert2Class=1),
            threshold: model.params.threshold,
            callback: callbackFn,
            epoch: 2000,
            learningRate: 0.01,
            verbose: 1
             } );

        // calcWeights.print()
        model.weights = calcWeights;
        return this;
    }

    this.classify = function(testDataX, weights=null, threshold=null, convert2Class = 1){
        // returns a one hot encoded vector
        const predClass = (this.logisticFn( threshold || model.params.threshold, convert2Class )( testDataX, weights || model.weights ));

        // model.weights.print();
        // predClass.print();
        // console.log(model.params.threshold,predProb)
        // converting prob to pred class

        // convert predClass to one-hot encoded vector.
        // predClass.matMul( predClass )
        const oneHot0 = tf.ones(predClass.shape).sub( predClass);
        const oneHotPredClass = oneHot0.concat( predClass, axis=1 );

        return oneHotPredClass;
    }
}
