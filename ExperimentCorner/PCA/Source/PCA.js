
function PCA(matrix){
    /*
        calculate its Covariance matrix
        calculate the SVD
     */

    // convert to ndArray
    const ndMatrix = nd.array(matrix.x);

    // calculating covariance matrix
    const covMatrix = nd.la.matmul(ndMatrix,ndMatrix.T);

    // decomposing covariance matrix using SVD
    const covSVD = nd.la.svd_decomp(covMatrix);

    
    // Extracting U S V matrixes
    const lBasis = covSVD[0];
    const sVals  = covSVD[1];
    const rBasis = covSVD[2];



    /* Visualization Using Plotly.js */



    // Visualizing the principle Component
    const pCompVec = lBasis.sliceElems('...',0);

    // Visualizing Data
    const data = [
        {
            x: convert2dArray( ndMatrix.sliceElems('...',0) ),
            y: convert2dArray( ndMatrix.sliceElems('...',1) ),
            mode: 'markers',
            type: 'scatter',
            marker: {size: 12}
        }
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

    // scree Plot
    const eigenValsPlot = [
    {
        y : convert2dArray(sVals),
        mode: 'lines+markers',
        type:'scatter',
        line: {color: "violet",width:3} ,
        markers: {width:10}

    }
    ];

    Plotly.newPlot('leftEigenVector',lVecData,{title: "U"});
    Plotly.newPlot('singularValues',sVecData,{title: "E"});
    Plotly.newPlot('rightEigenVector',rVecData,{title: "V"});

    Plotly.newPlot('screePlot',eigenValsPlot,{title: "SingularValues"});



    Plotly.plot('inputSpace',data,{title:'Data Space'});
    Plotly.plot('covMatrixPlot',covMatrixData,{title:'Covariance Matrix'});


}

