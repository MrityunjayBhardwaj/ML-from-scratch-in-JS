function LaplaceApproximation(){

    this.model = {
        fn : function() {return -1},
        params : {

        }
    },

    this.train = function(hessian,modeValue) {
       // given the hessian matrix this function will approximate/estimate the function.
        
       const M = hessian.shape[0];

       this.model.fn = function(z){
           const hessianDet = tfdet(hessian);
           
           const normalizationFac=  hessianDet.pow(1/2).div( Math.pow( 2*Math.PI, M/2 ));
           const quadraticExpression = (z.sub( modeValue)).matMul(hessian).matMul(z.sub(modeValue));

           return normalizationFac.matMul( tf.exp(quadraticExpression.mul(1/2) ));
       }


    },

    this.test = function(dataX){
        this.model.fn(dataX) ;
    }


}