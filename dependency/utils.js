let darkModeCols = {
  red: (alpha = 1) => `rgba(255, 99, 132, ${alpha})`,
  orange: (alpha = 1) => `rgba(255, 159, 64, ${alpha})`,
  yellow: (alpha = 1) => `rgba(255, 205, 86, ${alpha})`,
  green: (alpha = 1) => `rgba(75, 192, 192, ${alpha})`,
  blue: (alpha = 1) => `rgba(54, 162, 235, ${alpha})`,
  purple: (alpha = 1) => `rgba(153, 102, 255,${alpha})`,
  grey: (alpha = 1) => `rgba(231,233,237,  ${alpha})`,
  magenta: (alpha = 1) => `rgba(255,0,255,  ${alpha})`,
  violet: (alpha = 1) => `rgba(255,0,255,   ${alpha})`
};

function convert2dArray(ndArr) {
  // storing the shape of ndArray
  const shape = ndArr.shape;

  const nElems = shape[0];
  const nFeatures = shape[1] || 1;

  // fetch all the elements
  const arraySerialized = Array.from(ndArr.elems());

  // console.log
  const jsArray = Array(shape[0]);

  for (let i = 0; i < nElems; i++) {
    const cArrayRow = Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) {
      const index = i * nFeatures + j;

      const currVal = arraySerialized[index][1];
      if (nFeatures === 1) {
        jsArray[i] = typeof currVal === "object" ? currVal.re : currVal;
      } else {
        cArrayRow[j] = typeof currVal === "object" ? currVal.re : currVal;
      }
    }
    if (nFeatures !== 1) jsArray[i] = cArrayRow;
  }
  return jsArray;
}

function Img2Arr(img) {
  console.log("myIMG:", img);
  const imgWidth = img.width / 1;
  const imgHeight = img.height / 1;

  const imgArray = [];

  // const imgPixels = img.pixels;
  for (let i = 0; i < imgHeight; i++) {
    const imgRow = [];
    for (let j = 0; j < imgWidth; j += 1) {
      const index = i * imgWidth + j;

      imgRow.push(img.get(i, j)[0]);
    }

    imgArray.push(imgRow);
  }
  return imgArray;
}

/**
 *
 * @param {*} x
 * @param {*} y make sure they are one hot encoded.
 */
function classwiseDataSplit(x, y) {
  // make sure y is a one hot encoded vector.

  const nClasses = y.shape[1];

  const yArray = y.arraySync();
  const xArray = x.arraySync();

  const xSplit = [];
  for (let i = 0; i < nClasses; i++) {
    let currClassSplit = xArray.filter(function(_, index) {
      return this[index][i];
    }, yArray);
    xSplit.push(tf.tensor(currClassSplit));
  }

  return xSplit;
}

/**
 *
 * @param {*} x input must be a 'tf.tensor'
 */
function tf2nd(x) {
  // FIXIT:
  // if ( x instanceof tf )return nd.array ( x.arraySync() );
  if (true) return nd.array(x.arraySync());

  console.warn("the input is not tf.tensor");
  return x;
}

/**
 *
 * @param {*} x input must be an 'nd.array'
 */
function nd2tf(x) {
  // if ( x instanceof nd )return tf.tensor( convert2dArray( x ) );
  if (true) return tf.tensor(convert2dArray(x));

  console.warn("the input is not nd.array");
  return x;
}

function tfpinv(x) {
  return tf.tensor(pinv(x.arraySync()));
}

/**
 *
 * @param {*} x must be a javascript array
 * @description using moore-penrose psudoinverse procedure for calculating inverse of X
 * @returns return javascript Array of the pinv of X
 */
function pinv(x) {
  // convert js array to nd array
  const ndX = nd.array(x);

  // compute SVD NOTE: nd svd automatically remove the zero components in the singular matrix
  const xSVD = nd.la.svd_decomp(ndX);

  const { 0: lSVec, 1: singularVal, 2: rSVec } = xSVD;

  // compute the inverse of singularVec
  const singularDiag = nd.la.diag_mat(singularVal);

  const modDiag = singularDiag.forElems((val, i, j) => {
    if (i === j) singularDiag.set([i, j], 1 / val);
  });

  // now construct back the matrix in order to get our matrix psudo-inverse.
  const xInv = nd.la.matmul(lSVec, singularDiag, rSVec);

  return convert2dArray(xInv);
}

/**
 *
 * @param {*} x input must be a tf.tensor object where, shape[0] = no. of samples and shape[1] = no. of features
 * @param {*} meanCenter should input be mean centric before normalizing?
 * @returns normalized input matrix
 */
function normalize(x, meanCenter = 0) {
  // make the matrix mean centric.
  if (meanCenter) {
    const meanX = tf.mean(x, (axis = 0));
    x = x.sub(meanX);
  }

  // calculating the norm of all the axis
  let normVec = x.norm((axis = 1)).reshape([1, 1]);
  for (let i = 2; i < x.shape[1] + 1; i++) {
    normVec = normVec.concat(x.norm((axis = i)).reshape([1, 1]), (axis = 1));
  }

  // dividing x with its norm to get normalized x;
  return x.div(normVec);
}

function remap(n, start1, stop1, start2, stop2, withinBounds) {
  var newval = ((n - start1) / (stop1 - start1)) * (stop2 - start2) + start2;
  return newval;
}

function relativeMap(arrA, arrA_prime, arrB) {
  // this function takes arrA_prime and then replace arrA values with the corresponding arrB values assuming that B = f(A)

  /**
   * Example:-
   *      A = [1,2,3,4]
   *      B = [1,4,9,16] // here, B = (f(a) => a**2) i.e, they both somehow related to each other.
   *      A_prime = [0.0,0.5,1.0 ,1.5,2.0, 2.5,3.0, 3.5,4.0]
   *
   *      newArrA = [0.0,0.5,1.0 ,1.5,4.0, 2.5,9.0, 3.5,16.0]
   */

  let newArrA = arrA.slice(0);
  for (let i = 0; i < arrA_prime.length; i++) {
    let cAprimeVal = arrA_prime[i];
    let idxInArrA = arrA.indexOf(cAprimeVal);
    if (idxInArrA !== -1) newArrA[idxInArrA] = arrB[i];
  }
  return newArrA;
}

/**
 *
 * @param {*} A
 * @param {*} B
 * @param {*} A_sorted
 * @returns return the sorted B
 */
function sortWithIndexMap(B, indexMap) {
  // indexMap = [3,1,2]

  if (B.length != indexMap.length) {
    throw new Error(
      `Error: Invalid Inputs, \n inputs must be of same size,but given:\n B: ${
        B.length
      } and indexMap: ${indexMap.length}`
    );
    return false;
  }

  const B_sorted = Array(indexMap.length).fill(0);
  for (let i = 0; i < B_sorted.length; i++) {
    const g = indexMap[i];
    B_sorted[i] = B[g];

    // console.log(g,B_sorted[g]);
  }

  return B_sorted;
}

/**
 *
 * @param {Array} A javascript Array
 * @param {Array} B javascript Array
 * @description this function first sort array A and then sort B according to how the indexes of A changed when sorted
 * @example
 *  const A = [  5,  4,  8,  2, 1 ]
 *  const B = ['a','b','c','d','e']
 *  sortAB(A,B) // returns [ [1,2,4,5,8], ['e','d','b','a','c'] ]
 */
function sortAB(A, B) {
  const sortedIndexA = [];
  for (let i = 0; i < A.length; i++) {
    sortedIndexA.push(i);
  }

  const sortedA = A.slice();
  for (let i = 0; i < A.length; i++) {
    for (let j = i + 1; j < A.length; j++) {
      let cVal_i = sortedA[i];
      let cVal_j = sortedA[j];

      if (cVal_i > cVal_j) {
        // swap `em
        let foo = cVal_i;
        sortedA[i] = sortedA[j];
        sortedA[j] = foo;

        // swap the indexArray as well
        foo = sortedIndexA[i];
        sortedIndexA[i] = sortedIndexA[j];
        sortedIndexA[j] = foo;
      }
    }
  }
  // console.log(sortedIndexA,sortedA,A,Math.min(sortedA));
  return [sortedA, sortWithIndexMap(B, sortedIndexA)];
}

function prepro4Plotly(x) {
  if (x.constructor.toString().indexOf("Array") == -1)
    throw new Error('Error: Input Must Be of Type "Array"  ');

  return tf
    .tensor(x)
    .transpose()
    .reverse((axis = 0))
    .arraySync();
}

/**
 *
 * @param {object} range {x: {min: number, max: number}, y: {min: number, max: number}}
 * @param {number} division division b/w the range.
 */
function meshGridRange(
  range = { x: { min: 0, max: 1 }, y: { min: 0, max: 1 } },
  division
) {
  console.log(range);
  const a = tf
    .linspace(range.x.min, range.x.max, division)
    .flatten()
    .arraySync();
  const b = tf
    .linspace(range.y.min, range.y.max, division)
    .flatten()
    .arraySync();

  return meshGrid(a, b);
}

/**
 *
 * @param {Array} a
 * @param {Array} b
 * @description calculates the meshgrid just like in Matlab
 * @returns returns multidim js-array.
 */
function meshGrid(a, b) {
  // input must be an array
  // if(a.length !== b.length)throw new Error ( "Error: Input must be of same length.");

  const gridArray = [];
  for (let i = 0; i < a.length; i++) {
    const cRow = []; // current row
    const cVal_i = a[i];
    for (let j = 0; j < b.length; j++) {
      const cVal_j = b[j];
      cRow.push(cVal_j);
    }
    gridArray.push([cRow, Array(cRow.length).fill(cVal_i)]);
  }

  return gridArray;
}

/**
 *
 * @param {object} matrix input must be a tf.tensor object of shape NxM
 * @param {function } callbackFn callback function which modifies every element of the tensor object
 */
function tfMap(matrix, callbackFn) {
  const array = matrix.arraySync();
  const modArray = array.map(cRow => cVal => {
    callbackFn(cVal);
  });

  // return the modified tf.tensor
  return tf.tensor(modArray);
}

/**
 *
 * @param {object} vector input must be of shape Nx1 or 1xN
 * @return returns the sorted tf.tensor object
 */
function tfSort(vector) {
  if (vector.shape[1] > 1)
    throw new Error(" input must be of shape nx1 or 1xn.");
  const array = tensor.flattten().arraySync();

  const sortedArray = quickSort(array);

  // return the sorted tf.tensor;
  return tf.tensor(sortedArray);
}

function sleep(milliseconds) {
  var start = new Date().getTime();
  for (var i = 0; i < 1e7; i++) {
    if (new Date().getTime() - start > milliseconds) {
      break;
    }
  }
}

/**
 *
 * @param {string} type type of const function we require
 * @summary given the type this function returns a cost function which takes 3 params
 * data 'y' values and predicted 'y' value.
 */
function costFunction(type) {
  if (type === "mse") {
    // mean-squared-error (yPred - y)**2
    return function(y, yPred) {
      return tf.sum(tf.pow(tf.sub(yPred, y), 2));
    };
  }

  // add other cost function like R^2 etc.
}

/**
 *
 * @param {string} type type of const function we require
 * @summary given the type this function returns a cost function derivatives which takes 3 params
 * data x and y values and predicted 'y' value.
 */
function costFnDerivatives(type) {
  if (type === "mse") {
    return function(x, y, yPred) {
      // d/dx of MSE w.r.t 'w' : (yPred - y)x
      return tf.matMul(x.transpose(), tf.sub(yPred, y));
    };
  }
}

/**
 *
 * @param {object} x tf.tensor input data X
 * @param {object} y input data Y of type tf.tensor
 * @param {object} params important paramters for our optimizer
 * @param {function } params.yPridFn this function is provided with dataX and weights and returns the prediction Y
 * @param {function } params.costFn function which calculates the cost function given the data Y and predicted Y
 * @param {function}  params.costFnDx function which spits out the derivative of the cost function
 * @param {function}  params.callback a simple callback function which gets called at every epoch
 * @param {null}      params.yPred helpful for transfer learning
 * @param {object}    params.weights if you have pretrained weights vector, you can plug it in here, useful for transfer learning
 * @param {number}    params.epoch maximum number of gradient descent steps
 * @param {number}    params.learningRate
 * @param {number}    params.threshold the maximum amount allowable difference between prediction and truth.
 * @summary this function tries to find the weights which maximize/minimize the const function and using the parameters
 */
function optimize(
  x,
  y,
  params = {
    yPredFn: function(x, w) {
      return tf.matMul(w, x);
    },
    costFn: costFn("mse"),
    costFnDx: costFnDerivatives("mse"),
    callback: null,
    yPred: null,
    weights: null,
    epoch: 1000,
    learningRate: 0.001,
    threshold: 1e-3,
    verbose: false,
    batchSize: -1
  }
) {
  let {
    yPredFn = function(x, w) {
      return tf.matMul(w, x);
    },
    costFn = costFunction("mse"),
    costFnDx = costFnDerivatives("mse"),
    callback = null,
    yPred = null,
    weights = null,
    epoch = 100,
    learningRate = 0.0005,
    threshold = 1e-3,
    verbose = false,
    batchSize = -1
  } = params;

  // initializing weights vector tf.matMul( x, oldWeights).
  if (!weights) {
    // works only if x.shape = [m,n...] where, m == no. of training samples.
    weights = tf.randomNormal([x.shape[1], 1]);
  }

  let oldWeights = weights;
  for (let i = 0; i < epoch; i++) {
    const dataShuffleArray = x.concat(y, (axis = 1)).arraySync();
    tf.util.shuffle(Array);

    const cBatchX = tf
      .tensor(dataShuffleArray)
      .slice([0, 0], [-1, 2])
      .slice([0, 0], [batchSize, -1]);
    const cBatchY = tf
      .tensor(dataShuffleArray)
      .slice([0, 2], [-1, -1])
      .slice([0, 0], [batchSize, -1]);

    // calculating new prediction and loss function.
    yPred = yPredFn(cBatchX, oldWeights);
    const loss = costFn(cBatchY, yPred);

    if (verbose) {
      loss.print();
      oldWeights.print();
    }

    // checking convergence.
    if (loss.arraySync() < threshold) {
      yPred.print();
      return oldWeights;
    }

    // Calculating and Updating new Weights
    const weightDx = costFnDx(cBatchX, cBatchY, yPred, loss);
    const newWeights = tf.sub(
      oldWeights,
      tf.mul(tf.scalar(learningRate), weightDx)
    );

    // reAssigning weights
    oldWeights = newWeights;

    // invoke the callback function
    if (callback !== null) {
      console.log("inside Callback");
      callback(cBatchX, cBatchY, yPred, oldWeights, loss);
    }
  }

  return oldWeights;
}

/**
 *
 * @param {object} data input must be a tf tensor object
 */
function normalizeData(data) {
  return data.mean((axis = 0)).sub(data);
}

//NOTE: currently support only the [m,k] vector in [m,n] matrix where, k <= (start - n)
function insert2Tensor(originalTensor, insertTensor, start = [], end = []) {
  // const end = insertTensor.shape();
  // originalTensor.slice(start,end);

  const part1 = originalTensor.slice([0, 0], [-1, start[1]]);
  const part2 = originalTensor.slice(
    [0, start[1] + insertTensor.shape[1]],
    [-1, -1]
  );

  return part1.concat(insertTensor, (axis = 1)).concat(part2, (axis = 1));
}

// this function generates the span hyperplane given the Matrix/ basis vectors of column space
/**
 *
 * @param {object} A matrix of size [m,n]
 * @param {number} fac the size of the hyperplane
 * @summary this function construct a hyperplane using the basis vector of the column space of A.
 * the general useage of this method is to use it to visualize the span of the matrix 'A'.
 * @returns it returns the object inwhich {x : // x-compoents of all the sides of the hyperplane // ,y: //y-compnent// , z: // z- compenent // }, respectively.
 */
function genSpan(A, fac) {
  // first... convert it into an orthogonal matrix
  const Aortho = tf.linalg.gramSchmidt(A.transpose()).transpose();

  const p1 = Aortho.mul(fac);
  const p2 = Aortho.mul(-fac);

  // now connect p1 and p2 to form a hyperplane
  const combined = p1.concat(p2, (axis = 1));
  return {
    x: combined.slice([0, 0], [1, -1]),
    y:
      combined.shape[0] >= 2
        ? combined.slice([1, 0], [1, -1])
        : tf.zeros([1, combined.shape[0]]),
    z:
      combined.shape[0] === 3
        ? combined.slice([2, 0], [1, -1])
        : tf.zeros([1, combined.shape[0]])
  };
}


function replace2Tensor(originalTensor, replacedTensor, start = [0, 0]) {

  // check if the starting point is inside the originalTensor shape.
  if (start[0] < 0 && start[1] < 0 && start[0] > originalTensor.shape[0] && start[1] > originalTensor.shape[1] )new Error("invalid starting point specified");


  const part1_left  = originalTensor.slice(
    [0, 0],
    [-1, start[1]]
  );

  const part2_top   = originalTensor.slice(
    [0, start[1]],
    [start[0], (start[1] + replacedTensor.shape[1] <= originalTensor.shape[1] )?replacedTensor.shape[1] : -1 ]
  );

  const part4_right = ( 
    
    function(){

      if (start[1] + replacedTensor.shape[1] > originalTensor.shape[1])
        return tf.tensor([]);

      return originalTensor.slice(
        [0, start[1] + replacedTensor.shape[1]],
        [-1, -1]
      )
  
    }()
  );

  const part3_down = (
    function(){

      if (start[0] + replacedTensor.shape[0] > originalTensor.shape[0])
      return tf.tensor([]);
    
      return originalTensor.slice(
        [start[0] + replacedTensor.shape[0], start[1]],
        [-1, (start[1] + replacedTensor.shape[1] <= originalTensor.shape[1] )?replacedTensor.shape[1] : -1 ]
      );

    
    }()
  );

  // now putting all the parts together

  // tiling up part1_top, replacedTesor and part3_down
  let midSection = part2_top.concat(replacedTensor.slice([0,0],[-1,part2_top.shape[1]]), axis=0).concat(part3_down, axis=0);

  // console.log
  midSection = midSection.slice([0,0],[originalTensor.shape[0], -1]);

  const newTensor = part1_left.concat(midSection, axis=1).concat(part4_right, axis=1);

  return newTensor;
}


/**
 * 
 * @param {object} V  tf.tensor object of shape m by 1 which is essentially just a Vector
 * @summary given a vector this function convert it into diagonal matrix inwhich the diagonal entries corresponding to 
 * each values in the input vector 'V'.
 */
function tfDiag(V){
    if (!V.shape[1]){V = V.expandDims(1);}else{ if (V.shape[1] >1)throw new Error('input must be a Tf tensor of shape m by 1 but given m by n')}
    return V.mul(tf.eye(V.shape[0]));
}


function tfShuffle(V){

}

/**
 * 
 * @param {object} X input must be a tf.tensor
 * @param {object} Y input must be a tf.tensor and it must be a one hot encoded vector
 * @param {array} percent specify the percentage of train / test data spliting or if 2 values are specified then the second value corresponds to the additional cross-validation set split.
 */
function trainTestSplit(X, Y, percent){
  /**
   * TODO:-
   * 1. split the data according to there corresponding classes
   * 2. shuffle the data points
   * 3. take _percent_ data as a training data and rest of them as test data.
   * 4. if _percent.length()_ === 2 then split the test data into _percepnt[1]_ as cross-validation set.
   * 5. concat all the classwise X data for all the 3 sets (train,cross-validation and test)
   * 6. return all the 3 sets.
   */

   
  percent = (percent.length())? percent:[percent];// if the percent is just a number then convert it into array of length 1

  //  * 1. split the data according to there corresponding classes

  const classwiseX = classwiseDataSplit(X,Y);

  //  * 2. shuffle the data points

  for(let i=0; i<classwiseX.length(); i++){
    classwiseX[i] = tf.util.shuffle(classwiseX[i]);
  }

  //  * 3. take _percent_ data as a training data and rest of them as test data.

  const classwiseXTest = tf.tensor([]);
  const classwiseXCV   = tf.tensor([]);
  const classwiseXTest = tf.tensor([]);

  for(let i=0; i<classwiseX.length(); i++){

    const percentLength   = Math.floor( classwiseX[i].shape[0]*percent[0] );
    const percentCVLength = Math.floor( (classwiseX[i].shape[0])*(percent[1]) );

    classwiseXTrain = classwiseXTrain.concat( classwiseX[i].slice([0,0],[percentLength,-1]));

    const otherData = ( classwiseX[i].slice([percentLength,0],[-1,-1]) );

  //  * 4. if _percent.length()_ === 2 then split the test data into _percepnt[1]_ as cross-validation set.

    if (percent.length() === 2){
      classwiseXCV = classwiseXCV.concat( otherData.slice([0,0],[percentCVLength,-1]) );
      classwiseXTest = classwiseXTest.concat( classwiseX[i].slice([percentCVLength,0],[-1,-1]) );

      continue;
    }

    classwiseXTest = classwiseXTest.concat( classwiseX[i].slice([percentLength,0],[-1,-1]) );
  }

  //  * 6. return all the sets.

  const retVal = [classwiseXTest];

  if(percent.length() === 2)
    retVal.push(classwiseXCV);
  retVal.push(classwiseXTest);

  return retVal;
}