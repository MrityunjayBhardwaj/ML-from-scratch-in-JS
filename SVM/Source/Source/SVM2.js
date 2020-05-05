
function svm(){
    // private object 
    let model = {
        weights : tf.tensor([]),
        bias:tf.tensor([]),
        supportVectors: {x: tf.tensor([]),y: tf.tensor([]), alpha: tf.tensor([])},
        kernelType: 'linear',
        kernel: (x,y)=>{return x.matMul(y)},
        alpha: tf.tensor([]),
        data: {x: [], y: []}
    };

    let pairKernels = 0;
    let pairKernelsTF = 0;

    this.getParams = function(){
        return {weights: model.weights, bias: model.bias};
    };

    this.getAlphas = function(){
      return model.supportVectors.alpha;
      
    }
    this.getData = function(){
      return {x : model.supportVectors.x,y: model.supportVectors.y};
    }

    this.kernelFactory = function(type) {
        if (type === 'linear')return function(xI, xJ, params={}){
          return xI.reshape([xI.shape[0],1,xI.shape[1]]).matMul(xJ.reshape([xJ.shape[0], xJ.shape[1], 1])).reshape([xI.shape[0], 1])

        };
        if (type === 'poly')return function(xI, xJ, params={degree: 2}){
          let {degree = 2} = params;
          return xI.reshape([xI.shape[0],1,xI.shape[1]]).matMul(xJ.reshape([xJ.shape[0], xJ.shape[1], 1])).add(0).pow(degree).reshape([xI.shape[0], 1])
        };
        if(type ==='rbf') return function (xI,xJ, params={sigma: 0.5}){

          let {sigma =  0.5} = params;

          let s = tf.norm(xI.sub(xJ),ord=2,axis=1).pow(2);

          return tf.exp(s.mul(-1).div(2*sigma**2)).expandDims().transpose();
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

    this.fit = async (data, 
                      params = {
                                threshold: 0.00, 
                                tollerance: 0, 
                                epoch: 2, 
                                useSaved : false, 
                                kernelType: 'linear' | 'poly' | 'rbf' , 
                                kernelParams: {sigma: 0.5} 
                              } ) => {

      // user params:-
      let {
        threshold=0.01,
        tollerance = 1,// how much we want to tollerate the violation of our margin( '0' will make it hard margin )
        epoch = 2,
        verbose=false,
        tol = 1e-4,
        useSaved = true,
        kernelType = 'rbf',
        kernelParams = {sigma: 0.5}
      } = params;
        
      let N = data.x.shape[0];
      let C = tollerance;

      // initializing our dual variables
      let legrangeMultipliers = tf.zeros([data.x.shape[0], 1]);

      let weights = model.weights;
      let bias = model.bias;

      weights = tf.zeros([1, N]);
      bias = 0;

      // commiting params and legrangeMultipliers to our model object for further access
      model.weights = weights;
      model.bias = bias;

      let supportVectors = {x: data.x, y: data.y, alpha: legrangeMultipliers};
      model.supportVectors = supportVectors;

      model.kernelType = kernelType;
      model.kernel = this.kernelFactory(kernelType);
      const kernel = model.kernel;

      if(verbose)
        console.log('params: ', params, 'N: ', N)

      // fetching the saved data so that we don't have to convert our tensor data to arrays everytime
      let dX = model.data.x;
      let dY = model.data.y;

      let alpha = new Array(N).fill(0);
      model.alpha = alpha;

      if (!useSaved){

        if (verbose)
          console.log('not using saved!', useSaved);
        dX = data.x.arraySync();
        dY = data.y.arraySync();

        // assigning it to the global variable for future use.
        model.data.x = dX;
        model.data.y = dY;

        // pre-calc Kernel of data pairs ( save a ton of computation when only manipulating the same datasets)
        pairKernels = new Array(N);
        pairKernelsTF = new Array(N);
        for(let i=0;i<N;i++){
        let cDP = data.x.slice([i,0],[1, -1]).tile([N, 1]);
      
          pairKernelsTF[i] = kernel(cDP, data.x, kernelParams);
          pairKernels[i] = pairKernelsTF[i].dataSync();
        }

      }


      /* SMO algorithm */

      
      let passes = 0;// count the no of steps passes since we get no alpha updates

      for(let e=0; (e<epoch) && (passes < 10);e++){
        
        let alphaChanged = 0;
        for(let i=0;i<N;i++){

          let dataI = {x: dX[i],
                        y: dY[i][0],
                      };

          let alphaI = alpha[i];

            // residuals of our ith data point
          let ErrorI = this.marginOne(data.y, pairKernels[i]) - dataI.y

          if ( (dY[i]*(ErrorI) < -tol && alphaI < C) || (dY[i]*ErrorI > tol && alphaI > 0) ){

            // selecting our pair
            let j = i;
            while(j === i) j = Math.floor( Math.random()*N );

            let dataJ = {x: dX[j],
                          y: dY[j][0],
                        };

            // calculating the derivatives of objective function w.r.t our selected data points i and j
            let eta = 2*pairKernels[i][j] - pairKernels[i][i] - pairKernels[j][j];

            if (eta >= 0) continue;


            // residuals of our jth data point
            let ErrorJ = this.marginOne(data.y, pairKernels[j]) - (dataJ.y);

            let alphaJ = alpha[j];

            // calculating the lower and upper bound for our box constraints
            let lowerBound = 0; let upperBound = C;
            if(dataI.y === dataJ.y){
                lowerBound = Math.min(0, alphaI + alphaJ - C);
                upperBound = Math.max(C, alphaI + alphaJ);
            }
            else{
                lowerBound = Math.max(0, alphaJ - alphaI);
                upperBound = Math.min(C, alphaJ - alphaI + C);
            }

            if (tf.abs(lowerBound - upperBound)< .0001)continue;

            
            let newAlphaJ = alphaJ - dataJ.y*(ErrorI - ErrorJ)/eta;

            // forcing our alpha to remain inside our box constriant
            if (newAlphaJ > upperBound) newAlphaJ = upperBound;
            if (newAlphaJ < upperBound) newAlphaJ = lowerBound;
            if( Math.abs(alphaJ - newAlphaJ) < 0.0001)continue;

            let newAlphaI = alphaI + dataI.y*dataJ.y*(alphaJ - newAlphaJ);
         
            // calculating our bias
            let b1 = bias - ErrorI - pairKernels[i][i]*dataI.y*(newAlphaI - alphaI) - pairKernels[i][j]*dataJ.y*(newAlphaJ - alphaJ);
            let b2 = bias - ErrorJ - pairKernels[i][j]*dataI.y*(newAlphaI - alphaI) - pairKernels[j][j]*dataJ.y*(newAlphaJ - alphaJ);

            bias = (b1 + b2)/2;

            if (newAlphaI > 0 && newAlphaI < C) bias = b1;
            if (newAlphaJ > 0 && newAlphaJ < C) bias = b2;

            alpha[j] = newAlphaJ;
            alpha[i] = newAlphaI;

            alphaChanged++;

            // updating global variables
            model.alpha = alpha;
            // updating support vectors
            supportVectors = {x: data.x, y: data.y, alpha: tf.tensor(alpha).expandDims().transpose()};
            model.supportVectors = supportVectors;
            model.bias = bias;

              }

            
        }

        if(verbose)
          console.log('epoch:'+e+" alphaChanged: "+alphaChanged);

        if(alphaChanged === 0){
          passes++;
        }else passes = 0;

      }


      // collecting only those poits whose alphas > 0
      // let newSupportVectors = {x: tf.tensor([]), y: tf.tensor([]), alpha: tf.tensor([])};
      // for(let i=0;i< N;i++){
      //       let dataI = {x: data.x.slice([i,0], [1, -1]), 
      //                    y: data.y.slice([i,0], [1, -1])};
      //       let alphaI = supportVectors.alpha.slice([i,0], [1, -1]);
      //       if(alphaI.greater(0).dataSync()[0]){
      //           newSupportVectors.x = newSupportVectors.x.concat( dataI.x );
      //           newSupportVectors.y = newSupportVectors.y.concat( dataI.y );
      //           newSupportVectors.alpha = newSupportVectors.alpha.concat( alphaI );

      //       }
      // }


      let combinedData = data.x.concat(data.y,axis=1);

      combinedData = await tf.booleanMaskAsync(combinedData, supportVectors.alpha.greater(0).flatten())

      supportVectors.x = combinedData.slice([0,0],[-1, data.x.shape[1]]);
      supportVectors.y = combinedData.slice([0,data.x.shape[1]],[-1, -1]);
      supportVectors.alpha = tf.ones([combinedData.shape[0], 1]);

      combinedData.print();

      // // if(!useSaved)
      //   model.supportVectors = newSupportVectors;

      // console.log('final alphas: ', model.supportVectors.alpha.print())

//  if(this.kernelType === "linear") {

//         // compute weights and store them
//         this.w = new Array(this.D);
//         for(var j=0;j<this.D;j++) {
//           var s= 0.0;
//           for(var i=0;i<this.N;i++) {
//             s+= this.alpha[i] * labels[i] * data[i][j];
//           }
//           this.w[j] = s;
//           this.usew_ = true;
//         }
//       }

      if (kernelType === 'linear')
        model.weights = supportVectors.alpha.mul(supportVectors.y).mul(supportVectors.x).sum(axis=0).expandDims();

        console.log('final alphas: '+ supportVectors.alpha.print())

      return this;

      },

      this.marginOne = function (labels, preKernel){

        let f = model.bias;
        for(var i=0;i<labels.length;i++) {
          f += model.alpha[i] * labels[i][0] * preKernel[i];
        }

        return f;
      }

    this.margin = (data, preK /* precalculated kernel*/) =>{

      // if its a array then convert it into tf.tensor
        if (data[0]) data = tf.tensor(data).expandDims();


        if (data.shape[0] === 1){
        
           return model.supportVectors.alpha.mul(model.supportVectors.y)
            // .mul(model.kernel(model.supportVectors.x, data.tile([model.supportVectors.x.shape[0], 1]) ))
            .mul(preK || model.kernel(model.supportVectors.x, data.tile([model.supportVectors.x.shape[0], 1]) )) // use precalculated kernel values in order to save computation (if provided)
            .sum().add( model.bias );
        }

        if(model.kernelType === 'linear'){
          console.log(data.shape, model.weights.shape )
          return tf.matMul(data, model.weights.transpose() ).add(model.bias);
        }

        const nSupportVectors = model.supportVectors.x.shape[0];

        let output = model.supportVectors.alpha.slice([0, 0],[1, -1])
                    .mul(model.supportVectors.y.slice([0, 0], [1, -1]))
                    .mul(
                          model.kernel(
                            model.supportVectors.x.slice([0,0],[1,-1]).tile([data.shape[0], 1]), 
                            data
                          )
                        )
        for(let i=1;i<nSupportVectors;i++){

          const currOutput = model.supportVectors.alpha.slice([i, 0],[1, -1])
                            .mul(model.supportVectors.y.slice([i, 0], [1, -1]))
                            .mul(
                                model.kernel(
                                      model.supportVectors.x.slice([i,0],[1,-1]).tile([data.shape[0], 1]) , 
                                      data
                                  )
                                )

          // adding the influence of current support vector
          output = output.add(currOutput);
        }

        return output.add(model.bias);
    }
    this.test = (testDataX) =>{

        output = this.margin(testDataX);
        return output.greater(0).mul(2).sub(1);
    }



}
