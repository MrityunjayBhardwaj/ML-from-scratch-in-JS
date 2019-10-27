// SuperFast SVM


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

            const sqdist = tf.sum(x_i.pow(2),axis=1).expandDims(1).transpose()
                             .add( tf.sum(x_j.pow(2), axis=1).expandDims(1) )
                             .add( 
                                  tf.dot(x_i, x_j.transpose() ).transpose()
                                  .mul(-2)
                              );
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

    this.fit = (data, params = {threshold: 0.00, tollerance: 0, epoch: 2, }) => {

      // TODO:
      // randomizing the data for each epoch.


        function linearKernel(xI,xJ){
          // tackling the rectangular matrix
          // lets make xI.shape <= xJ alaways.. if it is not, then swap it
          if ((xJ.shape[0]) <  (xI.shape[0])){
            const tmp = xJ;
            xJ = xI;
            xI = tmp;

            console.log('xJ.shape[0] < xI.shape[0]');
          }
          // this trims the bigger matrix xJ to xI.shape and mimiking np.diag functionality
          xJ = xJ.slice([0, 0], [xI.shape[0], -1]); 
           return tfDiag( tf.matMul(xI, xJ, 0, 1)); 
        }

        function tfFilter(x, filter){

          const xArray = x.arraySync();
          filter = filter.flatten().arraySync();

          const filteredX = [];
          for(let i=0;i< xArray.length;i++){
            if(filter[i])filteredX.push(xArray[i]);
          }

          return tf.tensor(filteredX);
            
        }

        // user params:-
        let {
          threshold=0.01,
          tollerance = tf.tensor(.01).reshape([1, 1]),
          epoch = 2,
        } = params;
          

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

        data.x.print();
        data.y.print();

        // calculating weights from 15 


        let weights = model.weights;
        let bias = model.bias;


        weights = tf.sum( legrangeMultipliers.mul(data.y).mul(data.x), axis=0).expandDims();
        bias = tf.mean(
          data.y.sub( (weights.matMul(data.x.transpose()) ) )
        );

        weights.print();
        bias.print();


      // initializing for shuffling..
      let dataX = data.x;
      let dataY = data.y;

      const oldIndex = tf.linspace(0, data.x.shape[0]-1, data.x.shape[0]).expandDims(1);
      let shuffledIndex = oldIndex;


      for(let e=0;e<epoch;e++){
        /* shuffling the data. */

        // combining data.x, data.y, legrangeMultipliers and previousIndex 
        let combinedMatrix = dataX.concat(dataY, axis=1).concat(legrangeMultipliers, axis=1).concat( shuffledIndex, axis=1).arraySync();

        // shuffling the points and the dual variables randomly
        tf.util.shuffle(combinedMatrix);

        combinedMatrix = tf.tensor(combinedMatrix);

        const removedCombinedMatrix = combinedMatrix.slice([combinedMatrix.shape[0]-1, 0], [-1, -1]);

        // NOTE: here, pop one of the data points if data is odd numbered
        // because according to SMO the smallest pair we can optimize together is '2' so, later we will split our data points and variables into 2 groups
        // which if we have odd numbered data points then we have 1 extra point which doesn't have a corresponding point to make a pair
        // thats why, at each epoch we simply dropped one of the data point randomly and update the rest which stops us from any biased removal and overtime they converges...
        if ((data.x.shape[0] % 2) ){
          combinedMatrix = combinedMatrix.slice([0, 0], [combinedMatrix.shape[0] -1 , -1]);
        }

        console.log('combinedMatrix.Shape: '+ combinedMatrix.shape);

        // logging the starting and ending position every matrices inside our combinedMatrix.. for better referencing.
        const dataXColBlock =               [0                                                              , dataX.shape[1]];
        const dataYColBlock =               [dataXColBlock[1]                                               , dataY.shape[1]];
        const legrangeMultipliersColBlock = [dataYColBlock[1] + dataYColBlock[0]                            , legrangeMultipliers.shape[1]];
        const shuffledIndexColBlock =       [legrangeMultipliersColBlock[0] + legrangeMultipliersColBlock[1], shuffledIndex.shape[1]];

        // reassigning the shuffled matrices.
        dataX = combinedMatrix.slice([0, 0], [-1, dataXColBlock[1]]);
        dataY = combinedMatrix.slice([0, dataYColBlock[0]], [-1, dataYColBlock[1]]);
        legrangeMultipliers = combinedMatrix.slice([0, legrangeMultipliersColBlock[0]], [-1, legrangeMultipliersColBlock[1]]);
        shuffledIndex = combinedMatrix.slice([0,  shuffledIndexColBlock[0]], [-1, shuffledIndexColBlock[1] ]);


        /* splitting the data into 2 groups and each elem of both the group gets optimized jointly using tensor version of SMO algorithm */
        let x1 = dataX.slice([0,0],[Math.floor(dataX.shape[0]/2), -1]);
        let x2 = dataX.slice([Math.floor(dataX.shape[0]/2), 0],[-1, -1]);

        let y1 = dataY.slice([0,0],[Math.floor(dataY.shape[0]/2), -1]);
        let y2 = dataY.slice([Math.floor(dataY.shape[0]/2), 0],[-1, -1]);


        console.log("epoch: "+ e)

        let alpha1 = legrangeMultipliers.slice([0,0],[Math.floor(legrangeMultipliers.shape[0]/2), -1]);
        let alpha2 = legrangeMultipliers.slice([Math.floor(legrangeMultipliers.shape[0]/2), 0],[-1, -1]);

        let oldAlpha1 = alpha1;
        let oldAlpha2 = alpha2;

        const kernel = linearKernel;

        console.log('x1.shape', x1.shape, 'x2.shape', x2.shape)

        console.log(kernel(x1,x1).shape, kernel(x2,x2).shape, kernel(x1,x2).shape);
        // if all workes out we will get a tensor of size [n, 1]
        let objectiveFnDx = kernel(x1, x1).add(kernel(x2,x2)).sub(kernel(x1,x2).mul(2))

        const nonOptimalXFilter = objectiveFnDx.notEqual(0).mul(1);

        x1 = tfFilter(x1, nonOptimalXFilter);
        x2 = tfFilter(x2, nonOptimalXFilter);

        y1 = tfFilter(y1, nonOptimalXFilter);
        y2 = tfFilter(y2, nonOptimalXFilter);

        alpha1 = tfFilter(alpha1, nonOptimalXFilter);
        alpha2 = tfFilter(alpha2, nonOptimalXFilter);

        objectiveFnDx = tfFilter(objectiveFnDx, nonOptimalXFilter);

        // calc UpperBound:-
        const lB1 = tf.maximum(0, alpha2.sub(alpha1));
        const lB2 = tf.maximum(0, alpha2.add(alpha1)).sub(tollerance);
        const lB1Filter = y1.notEqual(y2).mul(1);
        const lB2Filter = y1.equal(y2).mul(1);
        const lB1Filtered = lB1.mul(lB1Filter);
        const lB2Filtered = lB2.mul(lB2Filter);

        const lowerBound = lB1Filtered.add(lB2Filtered);

        // calc LowerBound:-
        const uB1 = tf.minimum( tollerance, tollerance.sub((alpha1).add(alpha2)) );
        const uB2 =  tf.minimum(tollerance, alpha1.add(alpha2));
        const uB1Filter = y1.notEqual(y2).mul(1);
        const uB2Filter = y1.equal(y2).mul(1);
        const uB1Filtered = uB1.mul(uB1Filter);
        const uB2Filtered = uB2.mul(uB2Filter);

        const upperBound = uB1Filtered.add(uB2Filtered);

        const predY1 = weights.matMul(x1.transpose() ) .add(bias).transpose();
        const predY2 = weights.matMul(x2.transpose() ) .add(bias).transpose();

        const Error1 = predY1.sub(y1);
        const Error2 = predY2.sub(y2);

        // calculate alpha2 and use 19 we can get our alpha1
        let newAlpha2 = alpha2.add(
                          (y2.mul(Error1.sub(Error2))
                          .div(objectiveFnDx))
        );

    
        newAlpha2 = tf.maximum(newAlpha2, lowerBound);
        newAlpha2 = tf.minimum(newAlpha2, upperBound);


        // console.log('newAlpha2: ', newAlpha2.print());
        const s = y1.mul(y2);
        const newAlpha1 = alpha1.add(s.mul(newAlpha2.sub(alpha2)));


        let newLegrangeMultipliers = [];

        let newAlphaIndex = 0;
        let oldAlphaIndex = 0;
        // combining the optimal alphas with non-optimal new Alphas to form our full legrange multipliers matrix
        const nonOptimalFilterArray = nonOptimalXFilter.flatten().arraySync();
        for(let i=0; i< nonOptimalFilterArray.length; i++){
         
          const i2 = i+ dataX.shape[0]/2;
          if (nonOptimalFilterArray[i]){
            newLegrangeMultipliers[i]  = newAlpha1.flatten().arraySync()[newAlphaIndex];
            newLegrangeMultipliers[i2] = newAlpha2.flatten().arraySync()[newAlphaIndex];

            newAlphaIndex++;
          }else{
            newLegrangeMultipliers[i] = oldAlpha1.flatten().arraySync()[i]
            newLegrangeMultipliers[i2] = oldAlpha2.flatten().arraySync()[i];
          }

        }

        newLegrangeMultipliers = tf.tensor(newLegrangeMultipliers).expandDims(1);

        console.log('(newLegrangeMultipliers.shape): ',newLegrangeMultipliers.shape, nonOptimalFilterArray.length);


        // pop one of the data points if data is odd numbered
        if ((data.x.shape[0] % 2) ){

          console.log('yupp it is odd')
          // reassigning the shuffled matrices.
          const removedDataX = removedCombinedMatrix.slice([0, 0], [-1, dataXColBlock[1]]);
          const removedDataY = removedCombinedMatrix.slice([0, dataYColBlock[0]], [-1, dataYColBlock[1]]);
          const removedLegrangeMultipliers = removedCombinedMatrix.slice([0, legrangeMultipliersColBlock[0]], [-1, legrangeMultipliersColBlock[1]]);
          const removedShuffledIndex = removedCombinedMatrix.slice([0,  shuffledIndexColBlock[0]], [-1, shuffledIndexColBlock[1] ]);


          // reforming the matrices by adding the removed data point's matrices.
          dataX = dataX.concat(removedDataX, axis=0);
          dataY = dataY.concat(removedDataY, axis=0);
          newLegrangeMultipliers =  newLegrangeMultipliers.concat( removedLegrangeMultipliers, axis=0);
          legrangeMultipliers = legrangeMultipliers.concat(removedLegrangeMultipliers, axis=0);
          shuffledIndex = shuffledIndex.concat(removedShuffledIndex, axis=0);
        }

        console.log(dataX.shape);

        // updating weights and bias from this new Legrange multipliers..
        weights = tf.sum( newLegrangeMultipliers.mul(dataY).mul(dataX), axis=0).expandDims();
        bias = tf.mean(
          dataY.sub( (weights.matMul(dataX.transpose()) ) )
        );

      // calculating accuracy:-
        const updatedCalcPredY= this.hyperplaneFn(weights, bias);
        const predY = updatedCalcPredY(dataX);

        const correct = (
          tf.abs(
            tf.sum(
              tf.abs( 
                predY.greater(0) 
                .sub(dataY.greater(0))
              )
            )
          )
        .div(dataX.shape[0])
        .flatten().arraySync()[0]);
        console.log("accuracy: ", (correct) );

        const alphaDiff = (tf.sum(newLegrangeMultipliers.sub(legrangeMultipliers).pow(2)).flatten().arraySync()[0]);
        console.log("oldALphas - newAlphas: "+alphaDiff)

        // updating the legrange multipliers
        legrangeMultipliers = newLegrangeMultipliers;

        // check for convergence:-
        if (alphaDiff < threshold){
          
          console.log('legrange Multipliers Converges', newLegrangeMultipliers.flatten().arraySync());
          break;
        }


      }


        // commiting params and legrangeMultipliers to our model object for further access
        model.weights = weights;
        model.bias = bias;


        // unshuffle legrangeMultipliers :-
        const unShuffledLegrangeMultipliers = [];
        for(let i=0;i< legrangeMultipliers.shape[0]; i++){

          const currShuffledIndex = shuffledIndex.flatten().arraySync()[i];
          unShuffledLegrangeMultipliers[currShuffledIndex] = legrangeMultipliers.flatten().arraySync()[currShuffledIndex];
        }

        model.legrangeMultipliers = legrangeMultipliers;  


        return this;

      },

    this.test = (testData) =>{
        // return the prediction

        const {0: weights, 1: bias} = this.getParams();
        return weights.dot(testData.x.transpose()).add(bias).greater(0);
    }



}
