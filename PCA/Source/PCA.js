
function myPCA(){
  this.model = {
    covSVD : null

  }

  this.getExplainedVariance = function(){
    const nSamples = this.model.covSVD[0].shape[0]
    return this.model.covSVD[1].pow(2).div(nSamples -1);
  }

  this.fit = function(/* tf.tensor */ dataX ){

    const standardizeData = normalizeData(dataX, unitVariance=1);

    console.log("skdfj: ", standardizeData.print())

    // taking the svd of this covarience matrix after converting it to nd.array
    const covSVD = nd.la.svd_decomp( nd.array(standardizeData.arraySync()) );

    this.model.covSVD = [tf.tensor(convert2dArray(covSVD[0])),  
                         tfDiag(tf.tensor(convert2dArray(covSVD[1]))),  
                         tf.tensor(convert2dArray(covSVD[2]))];
   
    return this;
  }

  this.reconstruction = function(noOfComponents){

    // Extracting U S V matrixes
    const lVecs = this.model.covSVD[0];
    const sVals = this.model.covSVD[1];
    const rVecs = this.model.covSVD[2];

    if (noOfComponents > lVecs.shape[1])
      throw new Error('number of components must be less then'+lVecs.shape[1]);

    return lVecs.slice([0,0],[-1,noOfComponents])
           .matMul(sVals.slice([0,0],[noOfComponents, noOfComponents]))
           .matMul(rVecs.slice([0,0],[noOfComponents, -1]));

  }

  
}