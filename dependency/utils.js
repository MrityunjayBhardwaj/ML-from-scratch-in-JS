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