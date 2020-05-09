
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
    let pairKernelsTF = tf.tensor([]);

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

    this.calcObjectiveFn = (alpha, 
                            dataX, 
                            dataY,
                            kernelType='rbf', 
                            kernelParams={sigma: 0.5})=>{

      let N = dataX.shape[0];
      // calculating the kernel matrices
      let K = tf.tensor([]);

      // calculate the pairKernel if its not been calculated earlier

        for(let i=0;i<N;i++){
          let cDP = dataX.slice([i,0],[1, -1]).tile([N, 1]);
          K = K.concat( this.kernelFactory(kernelType)(cDP, dataX, kernelParams) );

        }

      K = K.reshape([N,N]);

      let YDiag = tfDiag(dataY);

      

      const mid = (YDiag)
                    .matMul(K)
                    .matMul(YDiag)

      const stackedMid =mid.expandDims().tile([alpha.shape[0],1,1]);

      let part1 =  alpha.reshape([alpha.shape[0], 1, alpha.shape[1]])
                    .matMul(stackedMid)
                    .matMul(alpha.reshape([alpha.shape[0], alpha.shape[1], 1]));

          part1 = part1.reshape([alpha.shape[0], 1]);

      const part2 = alpha.sum();

      const objFnOut = part1.mul(1/2).sub(part2);

      // calculating feasible regions

      let C = 1.0;

      let c1 = tf.clipByValue(alpha.greaterEqual(0).mul(alpha.lessEqual(C)).sum(axis=1), 0, 1);
      let c2 = dataY.transpose().tile([alpha.shape[0], 1]).mul(alpha).sum(axis=1).equal(0);

      const feasibleRegion = c2.mul(c1).expandDims().transpose();

      return {objFn: objFnOut, constraints: feasibleRegion};
    }

    this.fit = async (data, 
                      params = {
                                threshold: 0.00, 
                                tollerance: 0, 
                                epoch: 2, 
                                useSaved : false, 
                                kernelType: 'linear' | 'poly' | 'rbf' , 
                                kernelParams: {sigma: 0.5} ,
                                onFinishCallback : () => {console.log('finished')}
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
        pairKernelsTF = tf.tensor([]);
        for(let i=0;i<N;i++){
          let cDP = data.x.slice([i,0],[1, -1]).tile([N, 1]);
     
          const currKernelOutput =  kernel(cDP, data.x, kernelParams);
          pairKernelsTF = pairKernelsTF.concat(currKernelOutput);
          pairKernels[i] = currKernelOutput.dataSync();
        }

      }


      /* SMO algorithm */

      
      let passes = 0;// count the no of steps passes since we get no alpha updates

      // for diagnosis
    //   let js =[[6, 6, 9, 7, 8, 7, 5, 9, 5, 8],
    //   [2, 7, 9, 8, 8, 3, 0, 8, 1, 0],
    //   [3, 5, 9, 2, 7, 9, 1, 8, 1, 1],
    //   [3, 9, 8, 2, 7, 2, 9, 0, 1, 0],
    //   [5, 3, 8, 1, 8, 1, 5, 4, 1, 1],
    //   [7, 2, 5, 6, 9, 8, 7, 4, 0, 1],
    //   [4, 5, 0, 7, 8, 9, 0, 3, 1, 7],
    //   [1, 9, 8, 7, 8, 1, 4, 3, 5, 8],
    //   [6, 2, 4, 1, 9, 1, 3, 0, 9, 1],
    //   [9, 8, 4, 5, 2, 4, 4, 1, 9, 0]
    // ]
    //   epoch =10;

      let e = 0;


      // for(let e=0; (e<epoch) && (passes < 10);e++){

        let loopIntervel = await setInterval(() => {

        let alphaChanged = 0;
        for(let i=0;i<N;i++){

          let dataI = {x: dX[i],
                        y: dY[i][0],
                      };

          let alphaI = alpha[i];

            // residuals of our ith data point
          let ErrorI = this.marginOne( pairKernels[i]) - dataI.y

          if ( (dY[i]*(ErrorI) < -tol && alphaI < C) || (dY[i]*ErrorI > tol && alphaI > 0) ){

            // selecting our pair
            let j = i;
            while(j === i) j = Math.floor( Math.random()*N );
            // j = js[e][i];

            let dataJ = {x: dX[j],
                          y: dY[j][0],
                        };

            // calculating the derivatives of objective function w.r.t our selected data points i and j
            let eta = 2*pairKernels[i][j] - pairKernels[i][i] - pairKernels[j][j];

            if (eta >= 0) continue;


            // residuals of our jth data point
            let ErrorJ = this.marginOne(pairKernels[j]) - (dataJ.y);

            let alphaJ = alpha[j];

            // calculating the lower and upper bound for our box constraints
            let lowerBound = 0; let upperBound = C;
            if(dataI.y === dataJ.y){
                lowerBound = Math.max(0, alphaI + alphaJ - C);
                upperBound = Math.min(C, alphaI + alphaJ);
            }
            else{
                lowerBound = Math.max(0, alphaJ - alphaI);
                upperBound = Math.min(C, alphaJ - alphaI + C);
            }

            if (Math.abs(lowerBound - upperBound) < .0001)continue;

            
            let newAlphaJ = alphaJ - dataJ.y*(ErrorI - ErrorJ)/eta;

            // forcing our alpha to remain inside our box constriant
            if (newAlphaJ > upperBound) newAlphaJ = upperBound;
            if (newAlphaJ < lowerBound) newAlphaJ = lowerBound;
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

        // stopping criterion
        if (passes > 10 || e > epoch){
          clearInterval(loopIntervel);

          let combinedData = data.x.concat(data.y,axis=1).concat(supportVectors.alpha, axis=1);

          tf.booleanMaskAsync(combinedData, supportVectors.alpha.greater(0.0001).flatten()).then(
            (combinedData) =>{

          supportVectors.x = combinedData.slice([0,0],[-1, data.x.shape[1]]);
          supportVectors.y = combinedData.slice([0,data.x.shape[1]],[-1, 1]);
          supportVectors.alpha = combinedData.slice([0,combinedData.shape[1]-1], [-1, -1]);

              combinedData.print();

              if (kernelType === 'linear'){
                model.weights = supportVectors.alpha.mul(supportVectors.y).mul(supportVectors.x).sum(axis=0).expandDims();
              }

                console.log('final alphas: '+ supportVectors.alpha.print())


                if(params.onFinishCallback){params.onFinishCallback(this)}
            }


          )


        }

              if (kernelType === 'linear'){
                model.weights = supportVectors.alpha.mul(supportVectors.y).mul(supportVectors.x).sum(axis=0).expandDims();

              }
        if(params.onEpochCallback){params.onEpochCallback(this)}
        
        e++;
      }, 1000)
      // }

        
      },

      this.marginOne = function ( preKernel){

        let labels = model.data.y;

        let f = model.bias;
        for(var i=0;i<labels.length;i++) {
          f += model.alpha[i] * labels[i][0] * preKernel[i];
        }

        return f;
      }

    this.margin = (data ) =>{
        if(model.kernelType === 'linear'){
          console.log(data.shape, model.weights.shape )
          return tf.matMul(data, model.weights.transpose() ).add(model.bias);
        }

        const nSupportVectors = model.supportVectors.x.shape[0];

        let output = tf.tensor([]);
        if(nSupportVectors)
        output = model.supportVectors.alpha.slice([0, 0],[1, -1])
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
