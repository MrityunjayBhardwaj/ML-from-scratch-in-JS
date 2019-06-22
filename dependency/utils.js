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

    const nClasses = x.shape[1];

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