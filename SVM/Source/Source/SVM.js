
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

        function rbfKernel(xI, xJ, params={sigma: .5}){

          // let sigma = params.sigma || 0.5;
          let sigma =  0.5;

          let s = tf.norm(xI.sub(xJ),ord=2,axis=1).pow(2);

          return tf.exp(s.mul(-1).div(2*sigma**2)).expandDims().transpose();

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
          tollerance = tf.tensor(1).reshape([1, 1]), // how much you want to loose the "hard"-margin...
          epoch = 2,
          verbose=true,
          tol = 1e-4,
        } = params;
          

        tollerance = (typeof tollerance === 'number' )? tf.tensor(tollerance).reshape([1,1]) : tollerance;
           
        let legrangeMultipliers = tf.zeros([data.x.shape[0], 1]);

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
        // bias = tf.mean(
        //   data.y.sub( (weights.matMul(data.x.transpose()) ) )
        // );
        bias = tf.tensor([[0],])

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
      let supportVectors = {x: tf.tensor([]), y: tf.tensor([]), alpha: tf.tensor([])};

     
      for(let i=0;i< dataX.shape[0];i++){

        //   if (calcLegrangeMultipliers[i] > 0){
             supportVectors.x = supportVectors.x.concat(dataX.slice([i,0],[1,-1]));
             supportVectors.y = supportVectors.y.concat(dataY.slice([i,0],[1,-1]));
             supportVectors.alpha = supportVectors.alpha.concat( legrangeMultipliers.slice([i,0],[1,-1]) );
        //   }
      }

      model.supportVectors = supportVectors;

      model.kernel = rbfKernel;

      // const kernel = linearKernel;
      const kernel = model.kernel;

      let N = dataX.shape[0];

      let C = tollerance.dataSync()[0]

      let alphaArray = legrangeMultipliers.arraySync();

      // let dX = dataX.dataSync();
      // let dY = dataX.dataSync();
      // let alpha = legrangeMultipliers.dataSync();

      // let index2Use = [1,0,4,2,5,3]
      // let index2Use = [ 3, 0, 1, 2, 0, 1 ];
      let stopLoop = 0;

      let passes = 0;
      for(let e=0; (e<epoch) && (passes < 5);e++){

        let alphaChanged = 0;
        for(let i=0;i<N;i++){

            let dataI = {x: dataX.slice([i,0], [1, -1]), 
                         y: dataY.slice([i,0], [1, -1])};

            let alphaI = legrangeMultipliers.slice([i,0], [1, -1]);

            let ErrorI = this.margin(dataI.x).sub(dataI.y);

            if (   (dataI.y.mul(ErrorI).dataSync()[0] < -tol && alphaI.dataSync()[0] < C ) 
                || (dataI.y.mul(ErrorI).dataSync()[0] >  tol && alphaI.dataSync()[0] > 0 ) ){

                    let j = i;
                    while(j === i) j = Math.floor( Math.random()*N );

                    let dataJ = {x: dataX.slice([j,0], [1, -1]), 
                                 y: dataY.slice([j,0], [1, -1])};

                    let ErrorJ = this.margin(dataJ.x).sub(dataJ.y);

                    let alphaJ = legrangeMultipliers.slice([j,0], [1, -1]);

                    let lowerBound = 0; let upperBound = C;
                    if ( dataI.y.equal(dataJ.y).dataSync()[0]) {
                        lowerBound = tf.maximum(0, alphaI.add(alphaJ).sub(C));
                        upperBound = tf.minimum(C, alphaI.add(alphaJ));
                    }
                    else{
                        lowerBound = tf.maximum(0, alphaJ.sub(alphaI));
                        upperBound = tf.minimum(C, alphaJ.sub(alphaI).add(C));
                    }

                    if (tf.abs(lowerBound.sub(upperBound)).less(1e-4).dataSync()[0]) continue;

                    let eta = kernel(dataI.x, dataJ.x).mul(2).sub(kernel(dataI.x, dataI.x)).sub(kernel(dataJ.x, dataJ.x));

                    if (eta.greaterEqual(0).dataSync()[0]) continue;

                    let newAlphaJ = alphaJ.sub(dataJ.y.mul(ErrorI.sub(ErrorJ)).div(eta));

                    if (newAlphaJ.greater(upperBound).dataSync()[0]) newAlphaJ = upperBound;
                    if (newAlphaJ.less(lowerBound).dataSync()[0]) newAlphaJ = lowerBound;
                    if (tf.abs(alphaJ.sub(newAlphaJ)).less(1e-4).dataSync()[0]) continue;
                    
                    let newAlphaI = alphaI.add(dataI.y.mul(dataJ.y).mul(alphaJ.sub(newAlphaJ)));

                    // TODO: update the original alphas;
                   
                    let b1 = bias
                             .sub(ErrorI)
                             .sub(dataI.y.mul(newAlphaI.sub( alphaI))
                             .mul(kernel(dataI.x, dataI.x)))
                            .sub( dataJ.y.mul(newAlphaJ.sub(alphaJ)).mul(kernel(dataI.x, dataJ.x)) );

                    let b2 = bias
                             .sub(ErrorJ)
                             .sub(dataI.y.mul(newAlphaI.sub( alphaI))
                             .mul(kernel(dataI.x, dataJ.x)))
                            .sub( dataJ.y.mul(newAlphaJ.sub(alphaJ)).mul(kernel(dataJ.x, dataJ.x)) );

                    bias = b1.add(b2).mul(0.5);

                    if (newAlphaI.dataSync()[0] > 0 && newAlphaI.dataSync()[0] < C) bias = b1;
                    if (newAlphaJ.dataSync()[0] > 0 && newAlphaJ.dataSync()[0] < C) bias = b2;


                    alphaArray[j] = newAlphaJ.flatten().arraySync();
                    alphaArray[i] = newAlphaI.flatten().arraySync();

                    alphaChanged++;

                    legrangeMultipliers = tf.tensor(alphaArray);

                    // updating support vectors
                    supportVectors = {x: dataX, y: dataY, alpha: legrangeMultipliers};

                    model.supportVectors = supportVectors;

                    model.bias = bias;
                }

            
        }

        console.log('epoch:'+e+" alphaChanged: "+alphaChanged);
        if(alphaChanged === 0){
          passes++;
        }else passes = 0;

      }


      // collecting only those poits whose alphas > 0
      supportVectors = {x: tf.tensor([]), y: tf.tensor([]), alpha: tf.tensor([])};
      for(let i=0;i< N;i++){
            let dataI = {x: dataX.slice([i,0], [1, -1]), 
                         y: dataY.slice([i,0], [1, -1])};
            let alphaI = legrangeMultipliers.slice([i,0], [1, -1]);
            if(alphaI.greater(0).dataSync()[0]){
                supportVectors.x = supportVectors.x.concat( dataI.x );
                supportVectors.y = supportVectors.y.concat( dataI.y );
                supportVectors.alpha = supportVectors.alpha.concat( alphaI );

            }
      }

      model.supportVectors = supportVectors;

      console.log('final alphas: ', model.supportVectors.alpha.print())

      return this;

      },

    this.margin = (data) =>{

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
        // return the prediction



    // def predict_score(self, x):
    // return np.dot(self.alphas * self.sv_y, self._kernel(self.sv_x, x)) + self.b


        // const predScore =  tf.dot(modelParams.legrangeMultipliers.mul(modelParams.supportVectors.y), modelParams.kernel(modelParams.supportVectors.x, testDataX).add(modelParams.bias))

        output = this.margin(testDataX);

        return output.greater(0).mul(2).sub(1);
    }



}
