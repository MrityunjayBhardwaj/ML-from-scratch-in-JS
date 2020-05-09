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
        supportVectors: {x: tf.tensor([]),y: tf.tensor([])},
        kernel: (x,y)=>{return x.matMul(y)}
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
          // xJ = xJ.slice([0, 0], [xI.shape[0], -1]); 
           return tfDiag( tf.matMul(xI, xJ, 0, 1)); 
        }

       
        function polyKernel(xI,xJ, params={degree: 3}){
          if ((xJ.shape[0]) <  (xI.shape[0])){
            const tmp = xJ;
            xJ = xI;
            xI = tmp;

            console.log('xJ.shape[0] < xI.shape[0]');
          }
          let {degree = 10} = params;
          // return xI.mul(xJ).pow(degree).sum(1).expandDims().transpose()

          // return xI.matMul(xJ)

          return xI.reshape([xI.shape[0],1,xI.shape[1]]).matMul(xJ.reshape([xJ.shape[0], xJ.shape[1], 1])).add(0).pow(degree).reshape([xI.shape[0], 1])
        }

        function rbfKernel(xI, xJ, params={sigma: 8}){

          let sigma = params.sigma || 1.0;

          let s = tf.norm(xI.sub(xJ),ord=2,axis=1).pow(2);

          return tf.exp(s.mul(-1).div(sigma**2)).expandDims().transpose();

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
          tollerance = tf.tensor(.01).reshape([1, 1]), // how much you want to loose the "hard"-margin...
          epoch = 2,
          verbose=true,
        } = params;
          

        tollerance = (typeof tollerance === 'number' )? tf.tensor(tollerance).reshape([1,1]) : tollerance;
           
        let legrangeMultipliers = tf.randomUniform([data.x.shape[0], 1]).mul(2).sub(1);

        console.log("before Optimization:-", legrangeMultipliers.shape);
        legrangeMultipliers.print();

        if (verbose){
          console.log("input Data:-")
          data.x.print();
          data.y.print();

        }

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

        // commiting params and legrangeMultipliers to our model object for further access
        model.weights = weights;
        model.bias = bias;


      model.legrangeMultipliers = legrangeMultipliers;  

      const calcLegrangeMultipliers = legrangeMultipliers.flatten().arraySync();
      const supportVectors = {x: tf.tensor([]), y: tf.tensor([]), alpha: tf.tensor([])};

     
      for(let i=0;i< dataX.shape[0];i++){

          if (calcLegrangeMultipliers[i] > 0){
             supportVectors.x = supportVectors.x.concat(dataX.slice([i,0],[1,-1]));
             supportVectors.y = supportVectors.y.concat(dataY.slice([i,0],[1,-1]));
             supportVectors.alpha = supportVectors.alpha.concat( legrangeMultipliers.slice([i,0],[1,-1]) );
          }
      }

      model.supportVectors = supportVectors;

      model.kernel = rbfKernel;

      // const kernel = linearKernel;
      const kernel = model.kernel;

      let cont = 0;
      for(let e=0;e<epoch;e++){

        // combining data.x, data.y, legrangeMultipliers and previousIndex so that they can be manipulated together..
        let combinedMatrix = dataX.concat(dataY, axis=1).concat(legrangeMultipliers, axis=1).concat( shuffledIndex, axis=1).arraySync();

        // make sure that there will be no i == j
        let foundPerfectPairs = 0;

        // shuffling the points randomly
        tf.util.shuffle(combinedMatrix);


        combinedMatrix = tf.tensor(combinedMatrix);

        // copying the original combined matrix for later use..
        const originalCombinedMatrix = combinedMatrix.slice([0,0],[-1,-1]);

        // the data point that is not beeing considered in this epoch ( because, later we only use the n-1 data points if n is odd... in order to ease out the computation. )
        const removedCombinedMatrix = combinedMatrix.slice([combinedMatrix.shape[0]-1, 0], [-1, -1]);

        // NOTE: here, pop one of the data points if data is odd numbered
        // because according to SMO the smallest pair we can optimize together is '2' so, later we will split our data points and variables into 2 groups
        // which if we have odd numbered data points then we have 1 extra point which doesn't have a corresponding point to make a pair
        // thats why, at each epoch we simply dropped one of the data point randomly and update the rest which stops us from any biased removal and overtime they converges...
        if ((data.x.shape[0] % 2) ){
          combinedMatrix = combinedMatrix.slice([0, 0], [combinedMatrix.shape[0] -1 , -1]);
        }

        console.log('combinedMatrix.Shape: '+ combinedMatrix.shape);

        // logging the starting and ending position of every matrices inside our combinedMatrix.. for better referencing.
        const dataXColBlock =               [0                                                              , dataX.shape[1]];
        const dataYColBlock =               [dataXColBlock[1]                                               , dataY.shape[1]];
        const legrangeMultipliersColBlock = [dataYColBlock[1] + dataYColBlock[0]                            , legrangeMultipliers.shape[1]];
        const shuffledIndexColBlock =       [legrangeMultipliersColBlock[0] + legrangeMultipliersColBlock[1], shuffledIndex.shape[1]];

        // reassigning the matrices after shuffling...
        dataX = combinedMatrix.slice([0, 0], [-1, dataXColBlock[1]]);
        dataY = combinedMatrix.slice([0, dataYColBlock[0]], [-1, dataYColBlock[1]]);
        legrangeMultipliers = combinedMatrix.slice([0, legrangeMultipliersColBlock[0]], [-1, legrangeMultipliersColBlock[1]]);
        shuffledIndex = combinedMatrix.slice([0,  shuffledIndexColBlock[0]], [-1, shuffledIndexColBlock[1] ]);


        /* splitting the data into 2 random groups and each elem of both the group gets optimized jointly using tensor version of SMO algorithm */
        let x1 = dataX.slice([0,0],[Math.floor(dataX.shape[0]/2), -1]);
        let x2 = dataX.slice([Math.floor(dataX.shape[0]/2), 0],[-1, -1]);

        let y1 = dataY.slice([0,0],[Math.floor(dataY.shape[0]/2), -1]);
        let y2 = dataY.slice([Math.floor(dataY.shape[0]/2), 0],[-1, -1]);

        const predY1 = this.margin(x1);
        const predY2 = this.margin(x2);

        console.log("epoch: "+ e);

        let Error1 = predY1.sub(y1);
        let Error2 = predY2.sub(y2);

        const residual1 = Error1.mul(y1);
        const residual2 = Error2.mul(y2);

        let tol = 1e-4;

            



        // initializing alphas...
        let alpha1 = legrangeMultipliers.slice([0,0],[Math.floor(legrangeMultipliers.shape[0]/2), -1]);
        let alpha2 = legrangeMultipliers.slice([Math.floor(legrangeMultipliers.shape[0]/2), 0],[-1, -1]);

        let oldAlpha1 = alpha1;
        let oldAlpha2 = alpha2;

      // if( (labels[i]*Ei < -tol && this.alpha[i] < C) || (labels[i]*Ei > tol && this.alpha[i] > 0) )
        const booleanMask = residual1.less(-tol).mul(alpha1.less(tollerance)).add( residual1.greater(tol).mul(alpha1.greater(0)) );


        if (verbose){
          console.log('x1.shape', x1.shape, 'x2.shape', x2.shape);
          console.log(kernel(x1,x1).shape, kernel(x2,x2).shape, kernel(x1,x2).shape);
        }





        /* 
          Calculating LowerBound:-
          if the target y_1 == y_2 then use lB2 otherwise use lB1 
        */

        // calculating different lower bounds for both the cases..
        const lB1 = tf.maximum(0, alpha2.sub(alpha1));
        const lB2 = tf.maximum(0, alpha2.add(alpha1)).sub(tollerance);

        // using filters instead of conditional statements
        const lB1Filter = y1.notEqual(y2).mul(1);
        const lB2Filter = y1.equal(y2).mul(1);

        // applying filters...
        const lB1Filtered = lB1.mul(lB1Filter);
        const lB2Filtered = lB2.mul(lB2Filter);
        let lowerBound = lB1Filtered.add(lB2Filtered);

        /* 
          Calculating UpperBound:-
          if the target y_1 == y_2 then use uB2 otherwise use uB1 
        */

        // calculating different lower bounds for both the cases..
        const uB1 = tf.minimum( tollerance, tollerance.sub((alpha1).add(alpha2)) );
        const uB2 =  tf.minimum(tollerance, alpha1.add(alpha2));

        // using filters instead of conditional statements
        const uB1Filter = y1.notEqual(y2).mul(1);
        const uB2Filter = y1.equal(y2).mul(1);

        // applying filters...
        const uB1Filtered = uB1.mul(uB1Filter);
        const uB2Filtered = uB2.mul(uB2Filter);
        let upperBound = uB1Filtered.add(uB2Filtered);


        // if (Math.abs(L-H) < 1e-4)continue;
        const booleanMask2 = tf.abs(lowerBound.sub(upperBound)).greater(1e-4);


        // var eta = 2*this.kernelResult(i,j) - this.kernelResult(i,i) - this.kernelResult(j,j);

        // derivative of our quadratic optimization function..
        let objectiveFnDx = kernel(x1,x2).mul(2).sub( kernel(x1, x1)).sub(kernel(x2,x2))

        // if(eta >= 0) continue;
        const booleanMask3 = objectiveFnDx.less(0).mul(1);

        let combinedMask = booleanMask.mul(booleanMask2).mul(booleanMask3);


        if(combinedMask.sum().dataSync()[0] === 0){

          console.log('found all the legrange mulipliers', combinedMask.print())
          console.log('error:-', Error1.print(), Error2.print())
          model.supportVectors.alpha.print();
          // break;

        }

        /* filter out all the points and work with only those points that satisfies all 3 boolean Mask/conditionals */

        // filtering out those points and there corresponding alphas which are already being optimized.. (i.e, whose derivatives are already == 0)
        x1 = tfFilter(x1, combinedMask);
        x2 = tfFilter(x2, combinedMask);

        y1 = tfFilter(y1, combinedMask);
        y2 = tfFilter(y2, combinedMask);

        alpha1 = tfFilter(alpha1, combinedMask);
        alpha2 = tfFilter(alpha2, combinedMask);

        lowerBound = tfFilter(lowerBound, combinedMask);
        upperBound = tfFilter(upperBound, combinedMask);

        Error1 = tfFilter(Error1, combinedMask);
        Error2 = tfFilter(Error2, combinedMask);

        objectiveFnDx = tfFilter(objectiveFnDx, combinedMask);

        // // using only the non-optimal derivatives..
        // objectiveFnDx = tfFilter(objectiveFnDx, nonOptimalXFilter);


        // calculate alpha2 
        let newAlpha2 = alpha2.sub(
                          (y2.mul(Error1.sub(Error2))
                          .div(objectiveFnDx))
        );

    
        newAlpha2 = tf.maximum(newAlpha2, lowerBound); // if (newAlpha2 < lowerBound) newAlpha2 = lowerBound;
        newAlpha2 = tf.minimum(newAlpha2, upperBound); // if (newAlpha2 > upperBound) newAlpha2 = upperBound;

        // if(Math.abs(aj - newaj) < 1e-4) continue;
        let booleanMask4 = tf.abs(alpha2.sub(newAlpha2)).greater(1e-4);

        /* Combining booleanMask4 and original combineMask :- */
        const combinedMaskArray = combinedMask.flatten().arraySync();
        const bMask4Array = booleanMask4.flatten().arraySync();

        const newCombinedMask = combinedMaskArray.slice();

        let bMask4Index = 0;
        for(let i=0;i<combinedMaskArray.length;i++){

          if (combinedMaskArray[i]){
            newCombinedMask[i] = (bMask4Array[bMask4Index])
            bMask4Index++;
          }
        }

        combinedMask = tf.tensor(newCombinedMask).expandDims().transpose();


        console.log("combinedMask: ", combinedMask.print());

        x1 = tfFilter(x1, booleanMask4);
        x2 = tfFilter(x2, booleanMask4);

        y1 = tfFilter(y1, booleanMask4);
        y2 = tfFilter(y2, booleanMask4);

        alpha1 = tfFilter(alpha1, booleanMask4);
        alpha2 = tfFilter(alpha2, booleanMask4);

        newAlpha2 = tfFilter(newAlpha2, booleanMask4);

        Error1 = tfFilter(Error1, combinedMask);
        Error2 = tfFilter(Error2, combinedMask);

        // calculating the new alpha1 using the value of alpha2
        const s = y1.mul(y2);
        const newAlpha1 = alpha1.add(s.mul(newAlpha2.sub(alpha2)));


        console.log('printing alpha', newAlpha1.print(), newAlpha2.print())


        // update the bias term
        // var b1 = this.b - Ei - labels[i]*(newai-ai)*this.kernelResult(i,i)
        // - labels[j]*(newaj-aj)*this.kernelResult(i,j);

        // var b2 = this.b - Ej - labels[i]*(newai-ai)*this.kernelResult(i,j)
        //         - labels[j]*(newaj-aj)*this.kernelResult(j,j);

        // this.b = 0.5*(b1+b2);


        
        for(let i=0;i < Error1.shape[0];i++){
          b1 = bias.sub(Error1.slice([i, 0],[1, -1]))
               .sub( 
                 y1.slice([i, 0],[1, -1])
                 .mul(
                    newAlpha1.slice([i, 0],[1, -1])
                    .sub(alpha1.slice([i, 0],[1, -1]))
                    )
                 .mul(
                    kernel(x1.slice([i, 0],[1, -1]), x1.slice([i, 0],[1, -1])))
               ).sub(
                 y2.slice([i, 0],[1, -1])
                 .mul(
                      newAlpha2.slice([i, 0],[1, -1])
                      .sub(alpha2.slice([i, 0],[1, -1]))
                 ).mul(kernel(x1.slice([i, 0],[1, -1]),x2.slice([i, 0],[1, -1])))
                 );

          b2 = bias.sub(Error2.slice([i, 0],[1, -1]))
               .sub( 
                 y2.slice([i, 0],[1, -1])
                 .mul(
                    newAlpha1.slice([i, 0],[1, -1])
                    .sub(alpha1.slice([i, 0],[1, -1]))
                    )
                 .mul(
                    kernel(x1.slice([i, 0],[1, -1]), x2.slice([i, 0],[1, -1])))
               ).sub(
                 y2.slice([i, 0],[1, -1])
                 .mul(
                      newAlpha2.slice([i, 0],[1, -1])
                      .sub(alpha2.slice([i, 0],[1, -1]))
                 ).mul(kernel(x2.slice([i, 0],[1, -1]),x2.slice([i, 0],[1, -1])))
                 );

          bias = b1.add(b2).mul(0.5)

          const newa1 = newAlpha1.slice([i, 0],[1, -1]).arraySync()[0];
          const newa2 = newAlpha2.slice([i, 0],[1, -1]).arraySync()[0];

          if (newa1 > 0 && newa1 < tollerance )bias = b1;
          if (newa2 > 0 && newa2 < tollerance )bias = b1;
        }

        let newLegrangeMultipliers = [];

        let newAlphaIndex = 0;
        let oldAlphaIndex = 0;

        // combining the optimal alphas with non-optimal new Alphas to form our full legrange multipliers matrix
        const combinedBooleanMaskArray = combinedMask.mul(1).flatten().arraySync();
        for(let i=0; i< combinedBooleanMaskArray.length; i++){
         
          const i2 = i+ dataX.shape[0]/2;
          if (combinedBooleanMaskArray[i]){
            // update the new alphas for the non-optimal points..
            newLegrangeMultipliers[i]  = newAlpha1.flatten().arraySync()[newAlphaIndex];
            newLegrangeMultipliers[i2] = newAlpha2.flatten().arraySync()[newAlphaIndex];

            newAlphaIndex++;
          }else{
            // if the points is already optimized then leave it as it is..
            newLegrangeMultipliers[i] = oldAlpha1.flatten().arraySync()[i];
            newLegrangeMultipliers[i2] = oldAlpha2.flatten().arraySync()[i];
          }

        }

        newLegrangeMultipliers = tf.tensor(newLegrangeMultipliers).expandDims(1);

        // console.log('(newLegrangeMultipliers.shape): ',newLegrangeMultipliers.shape, nonOptimalFilterArray.length);


        // pop one of the data points if data is odd numbered (don't worry, we will use that point later in the next epoch {because these points are being chosen randomly})
        if ((data.x.shape[0] % 2) ){

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
        // weights = tf.sum( newLegrangeMultipliers.mul(dataY).mul(dataX), axis=0).expandDims();
        // bias = tf.mean(
        //   dataY.sub( (weights.matMul(dataX.transpose()) ) )
        // );

      // calculating accuracy:-
        // const updatedCalcPredY= this.hyperplaneFn(weights, bias);
        // const predY = updatedCalcPredY(dataX);

        // const correct = (
        //   predY.greater(0).equal(dataY.greater(0)).mul(1).sum()
        // .div(dataX.shape[0])
        // .flatten().arraySync()[0]);

        // console.log("accuracy: ", (correct) );

        // const alphaDiff = (tf.sum(newLegrangeMultipliers.sub(legrangeMultipliers).pow(2)).flatten().arraySync()[0]);
        // console.log("oldALphas - newAlphas: "+alphaDiff)

        // updating the legrange multipliers
        legrangeMultipliers = newLegrangeMultipliers;

        // updating bias:-
        model.bias = bias;


        // unshuffle the legrangeMultipliers :-
        // const unShuffledLegrangeMultipliers = [];
        // for(let i=0;i< legrangeMultipliers.shape[0]; i++){

        //   const currShuffledIndex = shuffledIndex.flatten().arraySync()[i];
        //   unShuffledLegrangeMultipliers[currShuffledIndex] = legrangeMultipliers.flatten().arraySync()[currShuffledIndex];
        // }

      model.legrangeMultipliers = legrangeMultipliers;  

      const calcLegrangeMultipliers = legrangeMultipliers.flatten().arraySync();
      const supportVectors = {x: tf.tensor([]), y: tf.tensor([]), alpha: tf.tensor([])};

     
      for(let i=0;i< dataX.shape[0];i++){

          if (calcLegrangeMultipliers[i] > 0){

             supportVectors.x = supportVectors.x.concat(dataX.slice([i,0],[1,-1]));
             supportVectors.y = supportVectors.y.concat(dataY.slice([i,0],[1,-1]));
             supportVectors.alpha = supportVectors.alpha.concat( legrangeMultipliers.slice([i,0],[1,-1]) );
          }
      }


      model.supportVectors = supportVectors;

      // model.kernel = polyKernel;


      }



      return this;

      },

    this.margin = (data) =>{

        const nSupportVectors = model.supportVectors.x.shape[0];

        let output = model.legrangeMultipliers.slice([0, 0],[1, -1])
                    .mul(model.supportVectors.y.slice([0, 0], [1, -1]))
                    .mul(
                          model.kernel(
                            model.supportVectors.x.slice([0,0],[1,-1]).tile([data.shape[0], 1]), 
                            data
                          )
                        )
        for(let i=1;i<nSupportVectors;i++){

          const currOutput = model.legrangeMultipliers.slice([i, 0],[1, -1])
                            .mul(model.supportVectors.y.slice([i, 0], [1, -1]))
                            .mul(
                                model.kernel(
                                      model.supportVectors.x.slice([0,0],[1,-1]).tile([data.shape[0], 1]) , 
                                      data
                                  )
                                )

          // adding the influence of current support vector
          output = output.add(currOutput);
        }

        return output.add(model.bias);
    }
    this.test = (testDataX) =>{
        // return the prediction



    // def predict_score(self, x):
    // return np.dot(self.alphas * self.sv_y, self._kernel(self.sv_x, x)) + self.b


        // const predScore =  tf.dot(modelParams.legrangeMultipliers.mul(modelParams.supportVectors.y), modelParams.kernel(modelParams.supportVectors.x, testDataX).add(modelParams.bias))

        output = this.margin(testDataX);

        return output.greater(0).mul(2).sub(1);
    }



}
