
/**
 * NOTE: Currently it only support 2 class classification.
 */
function svm(){
    // private object 
    let model = {
        weights : tf.tensor([]),
        bias:tf.tensor([]),
        legrangeMultipliers : tf.tensor([]),
    };

    this.getParams = function(){
        return {weights: model.weights, bias: model.bias};
    };

    this.getLegrangeMultipliers = function(){
      return model.legrangeMultipliers;
      
    }

    this.kernelFactory = function(type) {
        if (type === 'linear')return function(x_i, x_j, params={}){
          return tf.matMul( x_i, x_j.transpose());

        };
        if (type === 'quadratic')return function(x_i, x_j, params={}){
          return tf.dot(x_i, x_j.transpose()).pow(2);
        };
        if(type ==='rbf') return function (x_i,x_j, params={sigma: -1, smoothness : 1}){
            params.sigma = tf.tensor(1)|| params.sigma;

            const sqdist = tf.sum(x_i.pow(2),axis=1).expandDims(1).transpose().add( tf.sum(x_j.pow(2), axis=1).expandDims(1) ).add( tf.dot(x_i, x_j.transpose() ).transpose().mul(-2));
            return params.sigma.pow(2).mul( tf.exp( sqdist.mul( (-0.5/ (params.smoothness**2) ) )))

        }
    }


    // this function generate the equation for our hyperplane
    // and returns a function that take an arbitray value 'x'
    // which inturn spits out our predicted 'y' value -,+ or 0
    this.hyperplaneFn = function (weights, bias) {
      return function (x){
  
       return tf.matMul(x, weights.transpose() ).add(bias);

      }
    }

    this.fit = (data, params = {threshold: 0.01, tollerance: 1, epoch: 2, }) => {

        // TODO: 
        // make it fast using tensor computations..

        //DONE:
        // make it generalize for multi dimensions
        // add bias and update it
        // intialize alphas = 0
        // add linear kernel
        // add convergence criterion i.e, alphaOld - alphaNew < epsilon
        // check if y1.sub(y2).arraySync()[0] always gives true...
        // select alpha1 randomly


        // user params:-

        let {
          threshold=0.01,
          tollerance = tf.tensor(.01).reshape([1, 1]),
          epoch = 2,
        } = params;
         
        console.log(threshold, tollerance, epoch);

        tollerance = (typeof tollerance === 'number' )? tf.tensor(tollerance).reshape([1,1]) : tollerance;
           
      // initializing Legrange Multipliers
        let alphaArray = Array(data.x.shape[0]);


        for(let i=0;i< alphaArray.length;i++){

          // for now we have intialized it randomly:-
          alphaArray[i] = [(Math.random()*2 -1)*.00];

        }

        legrangeMultipliers = tf.tensor(alphaArray);

        console.log("before Optimization:-", legrangeMultipliers.shape);
        legrangeMultipliers.print();

        data.x.print()
        data.y.print()

        // calculating weights from 15 


        let weights = model.weights;
        let bias = model.bias;


        weights = tf.sum( legrangeMultipliers.mul(data.y).mul(data.x), axis=0).expandDims();
        bias = tf.mean(
          data.y.sub( (weights.matMul(data.x.transpose()) ) )
        );
        
        weights.print();
        bias.print();

        for(let e=0;e<epoch;e++){
          console.log("epoch: "+ e)


          // function for calculating PredY
          const calcPredY= this.hyperplaneFn(weights, bias);


          for(let i=0; i< data.x.shape[0]; i++){

            // choose 'j' s.t. j != i
            let j = Math.floor(Math.random()*data.x.shape[0])
            while (j === i){
              j = Math.floor(Math.random()*data.x.shape[0])
            }


            
            let alpha1 = legrangeMultipliers.slice([i, 0], [1, 1]);
            let alpha2 = legrangeMultipliers.slice([j, 0], [1, 1]);

            const x1 = data.x.slice([i, 0], [1, -1]);
            const x2 = data.x.slice([j, 0], [1, -1]);

            const y1 = data.y.slice([i, 0], [1, -1]);
            const y2 = data.y.slice([j, 0], [1, -1]);

            const kernel  = this.kernelFactory('linear');

            const objectiveFnDx = kernel(x1, x1).add(kernel(x2,x2)).sub(kernel(x1,x2).mul(2))

            //if we reach the minimum point for this pair of alpha then chose other alpha
            if (objectiveFnDx.flatten().arraySync()[0] === 0)continue;

            // clip the alphas which violates the box constraints:-
            const lowerBound = ( y1.sub(y2).flatten().arraySync()[0] ) ? 
              tf.maximum(0, alpha2.sub(alpha1)) :
              tf.maximum(0, alpha1.add(alpha2).sub(tollerance));

            const upperBound = ( y1.sub(y2).flatten().arraySync()[0] ) ?
              tf.minimum( tollerance, tollerance.sub((alpha1).add(alpha2)) ) :
              tf.minimum(tollerance, alpha1.add(alpha2));

            // calculating error for this data using old params
            
            const predY1 = calcPredY(x1);
            const predY2 = calcPredY(x2);

            const Error1 = predY1.sub(y1);
            const Error2 = predY2.sub(y2);

            // calculate alpha2 and use 19 we can get our alpha1
            let newAlpha2 = alpha2.add(
                              (y2.mul(Error1.sub(Error2)
                              .div(objectiveFnDx)))
            );


        
            // newAlpha2 = ( newAlpha2.arraySync()[0][0] >= lowerBound.arraySync()[0][0])? 
            //    newAlpha2 : lowerBound;

            console.log('newAlpha2, lowerBound, upperBound' );
            newAlpha2.print();
            lowerBound.print();
            upperBound.print();

            newAlpha2 = tf.maximum(newAlpha2, lowerBound);
            newAlpha2 = tf.minimum(newAlpha2, upperBound);

            // newAlpha2 = ( newAlpha2.arraySync()[0][0] <= upperBound.arraySync()[0][0])?
            //   newAlpha2 : upperBound;




            // calculating alpha1 using 23

            // console.log('newAlpha2: ', newAlpha2.print());
            const s = y1.mul(y2);
            const newAlpha1 = alpha1.add(s.mul(newAlpha2.sub(alpha2)));
            
            // commiting the new Alphas to our legrange Multiplier Tensor:-
            // legrangeMultipliers = insert2Tensor(legrangeMultipliers.transpose(), newAlpha1.concat(newAlpha2).transpose(), [0,i]).transpose();

            alphaArray[i] = newAlpha1.arraySync()[0]
            alphaArray[j] = newAlpha2.arraySync()[0]

          }
          
          newLegrangeMultipliers = tf.tensor(alphaArray);

          // calculating new Weights and biases from new dual variables
          weights = tf.sum( newLegrangeMultipliers.mul(data.y).mul(data.x), axis=0).expandDims();
          bias = tf.mean(
            data.y.sub( (weights.matMul(data.x.transpose()) ) )
          );

          // 0.2688176, -2.5751076 and bias: -.17151115159749847
          console.log('updated Weights:' ); 
          weights.print();

          console.log('updated bias:' ); 
          bias.print();

          // calculating accuracy:-
          const updatedCalcPredY= this.hyperplaneFn(weights, bias);
          const predY = updatedCalcPredY(data.x);

          const incorrect = (
            tf.abs(
              tf.sum(
                tf.abs( 
                  predY.greater(0) 
                  .sub(data.y.greater(0))
                )
              )
            )
          .div(data.x.shape[0])
          .flatten().arraySync()[0]);

          console.log("accuracy: ", (1-incorrect) );

          // checking for convergence:-
          const alphaDiff = (tf.sum(newLegrangeMultipliers.sub(legrangeMultipliers).pow(2)).flatten().arraySync()[0]);
          
          console.log("oldALphas - newAlphas: "+alphaDiff)
          tf.sum(newLegrangeMultipliers.sub(legrangeMultipliers).pow(2)).print()


          legrangeMultipliers = newLegrangeMultipliers;

          if (alphaDiff < threshold){
            
            console.log('legrange Multipliers Converges',newLegrangeMultipliers.flatten().arraySync());
            break;
          }



        }


        // commiting params and legrangeMultipliers to our model object for further access
        model.weights = weights;
        model.bias = bias;
        model.legrangeMultipliers = legrangeMultipliers;  

        return this;

      },

    this.test = (testData) =>{
        // return the prediction

        const {0: weights, 1: bias} = this.getParams();
        return weights.dot(testData.x.transpose()).add(bias).greater(0);
    }



}
