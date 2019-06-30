
function LeastSquares(){
    model = {
        weights : [],
        bias : []
    }

    this.train = function(data){

        // fetching the data.
        let A  = data.x;
        let y = data.y

        // initializing our design matrix 'A' and concatinating ones for intercepts.
        const ones = tf.ones([A.shape[0],1]);
        A = A.concat(ones,axis=1);


        // calculating A^T*A
        const p0 = tf.matMul( A.transpose(),A);
        const p1 = tf.matMul( A.transpose(),y);

        // (A^T * A )^-1
        const p0Inv = tf.tensor(pinv( p0.arraySync() ));

        const sol = tf.matMul(p0Inv, p1);

        // const luDecomp = nd.la.lu_decomp(p0p1Inv.arraySync());

        // const sol = nd.la.lu_solve( luDecomp[0],luDecomp[1],tf.zeros([A.shape[0],1]) );

        model.weights = sol.slice([0],[2]);
        model.bias    = sol.slice([sol.shape[0] - 1],[-1]);

        return this;
    }

    this.test = function(test){
        return tf.add( tf.matMul(test, model.weights), model.bias );
    }
}