function PCA(matrix){
    /*
        calculate its Covariance matrix
        calculate the SVD
     */

    // convert to ndArray
    let ndMatrix = nd.array(matrix.x);

    // normalizing the matrix
    ndMatrix = nd.array( [convert2dArray(ndMatrix.sliceElems('...',0)), convert2dArray(ndMatrix.sliceElems('...',1)) ]).T;

    // calculating mean
    let mean = nd.array(tf.fill(ndMatrix.shape,1/ndMatrix.shape[0]).arraySync());
    mean = nd.la.matmul(ndMatrix.T,mean).sliceElems('...',0);

    // making the ndMatrix mean centric
    ndMatrix.forElems( (elem,i,j) => ndMatrix.set([i,j] ,elem - mean((ndMatrix.shape[1]-1)*0+j) ) );

    // TEST:
    console.log(ndMatrix.T.shape,mean);
    window.mean = mean;
    window.mtx = ndMatrix;


    console.log(ndMatrix)
    // calculating covariance matrix
    const covMatrix = nd.la.matmul(ndMatrix.T,ndMatrix);
    
    // decomposing covariance matrix using SVD
    const covSVD = nd.la.svd_decomp(covMatrix);

    // TODO: Normalize covariance Matrix in order to prevent having to normalize eigenvals and vecs.
    // decomposing covriance matrix using eigenDecomposition (which is preferred.)
    const eigenDecomp = nd.la.eigen(covMatrix);

    console.log(eigenDecomp[0], convert2dArray(eigenDecomp[1]));
    
    // Extracting U S V matrixes
    const lBasis = covSVD[0];
    const sVals  = covSVD[1];
    const rBasis = covSVD[2];

    /* Visualization Using Plotly.js */


    // Visualizing the principle Component
    // const pCompVec = convert2dArray(nd.la.matmul(lBasis.sliceElems('...',0).reshape(1,2),ndMatrix.T ));
    const pCompVec = convert2dArray(eigenDecomp[1]);

    const eVecFac = 2; // multiply the eigenVectors for clear visualization.

    let eigenVec = eigenDecomp[1];
    let eigenVals  = convert2dArray(eigenDecomp[0]);

    // normalizing eigenValues.
    let eValMag = (eigenVals[0] +eigenVals[1]);
    eigenVals[0] = eigenVals[0]/eValMag;
    eigenVals[1] = eigenVals[1]/eValMag;

    console.log(pCompVec[0],covMatrix,eigenVals);


var layout = {
  showlegend: false,
  autosize: false,
  width: 600,
  height: 550,
  margin: {t: 50},
  hovermode: 'closest',
  bargap: 0,
  xaxis: {
    domain: [0, 0.85],
    showgrid: false,
    zeroline: false
  },
  yaxis: {
    domain: [0, 0.85],
    showgrid: false,
    zeroline: false
  },
  xaxis2: {
    domain: [0.85, 1],
    showgrid: false,
    zeroline: false
  },
  yaxis2: {
    domain: [0.85, 1],
    showgrid: false,
    zeroline: false
  }
};
    // Visualizing Data
    const data = [
        {
            x: convert2dArray( ndMatrix.sliceElems('...',0) ),
            y: convert2dArray( ndMatrix.sliceElems('...',1) ),
            mode: 'markers',
            type: 'scatter',
            marker: {size: 20}
        },
        {
            x : [0, pCompVec[0][0]*eigenVals[0]**1*eVecFac],
            y : [0, pCompVec[1][0]*eigenVals[0]**1*eVecFac],
            mode: 'Lines',
            type: 'Lines',
            line: {width: 5,color:'violet'},
        },
        {
            x : [0, pCompVec[0][1]*eigenVals[1]**1*eVecFac],
            y : [0, pCompVec[1][1]*eigenVals[1]**1*eVecFac],
            mode: 'Lines',
            type: 'Lines',
            line: {width: 5,color:'yellow'},
        },


    ];

    // Visualizing Covariance Matrix
    const covMatrixData = [
        {
            z: convert2dArray(covMatrix),
            type: 'heatmap'
        }
    ]

    // Visualizing all SVD components
    
    var lVecData = [
    {
        z:convert2dArray(lBasis),
        type: 'heatmap'
    },
    
    ];
    var sVecData = [
    {
        z:convert2dArray(nd.la.diag_mat( sVals)),
        type: 'heatmap'
    },
    
    ];

    var rVecData = [
    {
        z:convert2dArray(rBasis),
        type: 'heatmap'
    },
    
    ];

    // Visualizing Eigen Vectors using scree Plot
    const eigenValsPlot = [
    {
        y : convert2dArray(sVals),
        mode: 'lines+markers',
        type:'scatter',
        line: {color: "violet",width:3} ,
        markers: {width:10}

    }
    ];

    // calling plotly for creating new plots

    Plotly.newPlot('leftEigenVector',lVecData,{title: "U"});
    Plotly.newPlot('singularValues',sVecData,{title: "E"});
    Plotly.newPlot('rightEigenVector',rVecData,{title: "V"});

    Plotly.newPlot('screePlot',eigenValsPlot,{title: "SingularValues"});

    Plotly.react('inputSpace',data,layout);
    Plotly.react('covMatrixPlot',covMatrixData,{title:'Covariance Matrix'});

    // Calculating Covariance Surface
    const res = 15;
    const w = tf.linspace(-2,2,res);
    const tfCovMatrix = tf.tensor(convert2dArray(covMatrix));

    // let covSurf = [];
    // for(let i=0;i<res;i++){
    //     const cW_i = w.slice(i,1);
    //     const currSurfRow = new Array(res);
    //     for(let j=0;j<res;j++){
    //     const cW_j = w.slice(j,1);
    //         const cW = tf.tensor([cW_i.arraySync(),cW_j.arraySync()]).transpose();
    //         currSurfRow[j] = tf.matMul(tf.matMul(cW,tfCovMatrix),cW.transpose()).arraySync()[0][0] / tf.matMul(cW,cW.transpose()).arraySync()[0][0];
    //     }
    //     covSurf.push(currSurfRow)
    // }

    // console.log(covSurf);

  // Visualizing Covariance Surfaces:
  // var covSurfData = [{
  //           z: covSurf,
  //           type: 'surface'
  //         }];
    
  // var covSurfLayout = {
  //   title: 'Covariance Surface',
  //   autosize: false,
  //   width: 500,
  //   height: 500,
  // };

  // Plotly.newPlot('covSurfViz', covSurfData, covSurfLayout);

// Karuna-Loeve transform : Projecting z-score of the data onto the eigenvectors(new basis)

// Calculating Z-score
const xMatrix = tf.tensor(matrix.x);

const originalShape =  xMatrix.shape;

// coverting to vector:
const xVector = xMatrix.reshape([xMatrix.shape[0]*xMatrix.shape[1],1]);

const xVecMean  = xVector.mean();
const xVecStdev = tf.sum( tf.pow(tf.pow(xVector.sub(xVecMean),2),1/2 ) );

// coverting to ndArray
const zScoreX = nd.array(tf.div( tf.sub(xMatrix,xVecMean) , xVecStdev ).arraySync());

// taking the subset of the eigenVectors (for compression)

// T = Z*E;
const KLT = ( nd.la.matmul(zScoreX,covMatrix) );


console.log(KLT);

// Visualizing KLT
// TODO: fix reconstruction  DONE: 
// TODO: find why z-score method doesn't work

const p = 2 ;
const eigenVec02p = eigenVec.sliceElems('...',[0,p]);


console.log(nd.array(matrix.x))
const xHat = nd.la.matmul(nd.la.matmul(eigenVec02p,eigenVec02p.T),nd.array(matrix.x).T).T;

window.eVec = eigenVec;
const KLTData = [{
  x: convert2dArray( xHat.sliceElems('...',0) ),
  y: convert2dArray( xHat.sliceElems('...',1) ),
  mode: 'markers',
  type: 'scatter',
  markers: {width: 2}
}];

Plotly.newPlot('KLT',KLTData,{title: 'K-Loeve Transform'})
}