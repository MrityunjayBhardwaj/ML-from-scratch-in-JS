function convert2dArray(ndArr){
  // storing the shape of ndArray
  const shape = ndArr.shape;

  const nElems=  shape[0];
  const nFeatures = shape[1] || 1  ; 

  // fetch all the elements
  const arraySerialized = Array.from(ndArr.elems());

  // console.log
  const jsArray = Array(shape[0]);

  for(let i=0;i<nElems;i++){

    const cArrayRow = Array(nFeatures);
    for(let j=0;j<nFeatures;j++){
      const index = i*nFeatures + j;

      const currVal = arraySerialized[index][1];
      if (nFeatures === 1 ){
        jsArray[i] = ( typeof currVal === 'object')? currVal.re : currVal;
      }
      else{
        cArrayRow[j] =  ( typeof currVal === 'object')? currVal.re : currVal;
      }

    }
    if(nFeatures !== 1)
    jsArray[i] = cArrayRow;
  }
  return jsArray;
}

function Img2Arr(img){
  console.log("myIMG:",img);
  const imgWidth  = img.width/1;
  const imgHeight = img.height/1;

  const imgArray = [];

  // const imgPixels = img.pixels;
  for(let i=0;i<imgHeight;i++){

    const imgRow = []
    for(let j = 0;j<imgWidth;j += 1){
       const index =  i*imgWidth + j;

      imgRow.push(img.get(i,j)[0]);
    }

    imgArray.push(imgRow)
  }
return imgArray;
}

/**
 * 
 * @param {*} x 
 * @param {*} y make sure they are one hot encoded.
 */
function classwiseDataSplit(x,y){
    // make sure y is a one hot encoded vector.

    const nClasses = y.shape[1];

    const yArray = y.arraySync();
    const xArray = x.arraySync();

    const xSplit = []
    for(let i=0;i<nClasses;i++){
        let currClassSplit = xArray.filter(function(_,index){ return this[index][i]; },yArray);
        xSplit.push( tf.tensor( currClassSplit ) );
    }

    return xSplit; 
    
}

/**
 * 
 * @param {*} x input must be a 'tf.tensor'
 */
function tf2nd(x){
  // FIXIT:
  // if ( x instanceof tf )return nd.array ( x.arraySync() );
  if(true)return nd.array ( x.arraySync() );

  console.warn('the input is not tf.tensor');
  return x;
}

/**
 * 
 * @param {*} x input must be an 'nd.array'
 */
function nd2tf(x){
  // if ( x instanceof nd )return tf.tensor( convert2dArray( x ) );
  if ( true )return tf.tensor( convert2dArray( x ) );

  console.warn('the input is not nd.array');
  return x;
}

/**
 * 
 * @param {*} x must be a javascript array
 * @description using moore-penrose psudoinverse procedure for calculating inverse of X
 * @returns return javascript Array of the pinv of X
 */
function pinv(x){

  // convert js array to nd array
  const ndX = nd.array(x);

  // compute SVD NOTE: nd svd automatically remove the zero components in the singular matrix
  const xSVD = nd.la.svd_decomp(ndX);

  const {0:lSVec , 1: singularVal , 2: rSVec} = xSVD;

  // compute the inverse of singularVec
  const singularDiag = nd.la.diag_mat(singularVal);

  const modDiag = singularDiag.forElems( (val,i,j) =>{ if (i===j)singularDiag.set([i,j], 1/val) });


  // now construct back the matrix in order to get our matrix psudo-inverse.
  const xInv = nd.la.matmul( lSVec, singularDiag, rSVec );

  return convert2dArray(xInv);
}

/**
 * 
 * @param {*} x input must be a tf.tensor object where, shape[0] = no. of samples and shape[1] = no. of features
 * @param {*} meanCenter should input be mean centric before normalizing?
 * @returns normalized input matrix
 */
function normalize(x,meanCenter=0){

  // make the matrix mean centric.
  if (meanCenter){
    const meanX = tf.mean(x,axis=0);
    x = x.sub(meanX);
  }

  // calculating the norm of all the axis 
  let normVec =( x.norm(axis=1).reshape([1,1]) )
  for(let i=2;i< x.shape[1]+1;i++){
    normVec = normVec.concat( x.norm(axis=i).reshape([1,1]) ,axis=1)
  }

  // dividing x with its norm to get normalized x;
  return x.div(normVec);
}

function remap(n, start1, stop1, start2, stop2, withinBounds) {
  var newval = (n - start1) / (stop1 - start1) * (stop2 - start2) + start2;
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
          if (idxInArrA !== -1)
              newArrA[idxInArrA] = arrB[i];
      }
      return newArrA;
};

/**
 * 
 * @param {*} A 
 * @param {*} B 
 * @param {*} A_sorted 
 * @returns return the sorted B
 */
function sortWithIndexMap(B,indexMap){
  // indexMap = [3,1,2]

  if (B.length != indexMap.length){
    throw new Error(`Error: Invalid Inputs, \n inputs must be of same size,but given:\n B: ${B.length} and indexMap: ${indexMap.length}`) ;
    return false;}

  const B_sorted =  Array(indexMap.length).fill(0);
  for(let i = 0; i<B_sorted.length;i++){
    const g = indexMap[i];
    B_sorted[g] = B[i];
  }

  return B_sorted;
}

function sortAB(A,B){

  const sortedIndexA = [];
  for(let i=0;i<A.length;i++){
    sortedIndexA.push(i);
  };

  const sortedA = A;
  for(let i=0;i<A.length;i++){
    let cVal_i = sortedA[i];
    for(let j=i;j<A.length ;j++){
    let cVal_j = sortedA[j];
      if (cVal_i > cVal_j){
        // swap `em
        let foo =  cVal_i;
        cVal_i  = cVal_j;
        cVal_j  = foo;

        // swap the indexArray as well
        foo = sortedIndexA[i];
        sortedIndexA[i] = sortedIndexA[j];
        sortedIndexA[j] = foo;
      }
    }
  }
  return sortWithIndexMap(B,sortedIndexA);
}

function prepro4Plotly(x){
  if ( x.constructor.toString().indexOf("Array") == -1 )throw new Error("Error: Input Must Be of Type \"Array\"  " );

  return tf.tensor(x).transpose().reverse(axis=0).arraySync();
}

/**
 * 
 * @param {Array} a 
 * @param {Array} b 
 * @description calculates the meshgrid just like in Matlab
 */
function meshGrid(a,b){
 // input must be an array
if(a.length !== b.length)throw new Error ( "Error: Input must be of same length.");

 const gridArray = [];
 for(let i=0;i<a.length;i++){
   const cRow = [];// current row
   const cVal_i = a[i];
   for(let j=0;j<b.length;j++){
     const cVal_j = b[j];
    cRow.push( cVal_j );
   }
   gridArray.push( [cRow, Array(cRow.length).fill(cVal_i) ])
 }
 return gridArray;
}