

function SVD(matrix ){

// infering some image properties.
const imageDim = {width: matrix[0].length, height: matrix.length};
const imageDimRatio = {width: imageDim.width/(imageDim.width+imageDim.height), height: imageDim.height/(imageDim.width+imageDim.height)};

  const data = [
    {
      z: prepro4Plotly(matrix),
      colorscale: 'Greys',
      type:'heatmap',
    }
  ];
// initializing plot
Plotly.newPlot('tester', data,{title:'Original Image',

        font : {
            size : 15,
            color: 'white',
            family : 'Helvetica'
        },
        paper_bgcolor : '#222633',

},
);


// calculating the svd of this image.
const matrixSvd = nd.la.svd_decomp(matrix);

const matrixLEVec =  convert2dArray(matrixSvd[0]);
const matrixSEVal =  convert2dArray(nd.la.diag_mat(matrixSvd[1]));
const matrixREVec =  convert2dArray(matrixSvd[2]);

// matrixSEVal[0][0] = -matrixSEVal[0][0]*2;

// console.log("matrixSEVal" ,matrixSEVal);

// converting the U S V to tf.tensor for further calcuation
let tfU = tf.tensor(matrixLEVec);
let tfS = tf.tensor(matrixSEVal);
let tfV = tf.tensor(matrixREVec);

// tfU = tfU.pow(1/2)
// tfS = tfS.pow(-1)
// tfV = tfV.pow(1/2)


 // slice the left eigen vector matrix, just for experimentation
const tfUMod = tf.mul(tfU.slice([0,0],[-1,5]),0).concat(tfU.slice([0,5],[-1,-1]), 1);

// preparing the data for heatmap
const eigenVals = convert2dArray(matrixSvd[1]); // eigen value matrix
var lVecData = [
  {
    // z: [[1, 20, 30, 30], [20, 1, 60, 50], [30, 60, 1, 70], [0, 30, 60,8]],
    z: prepro4Plotly( matrixLEVec ),
    type: 'heatmap',
  },
  
];
var sVecData = [
  {
    // z: [[1, 20, 30, 30], [20, 1, 60, 50], [30, 60, 1, 70], [0, 30, 60,8]],
    z: prepro4Plotly( matrixSEVal ),
    type: 'heatmap',
  },
  
];

var rVecData = [
  {
    // z: [[1, 20, 30, 30], [20, 1, 60, 50], [30, 60, 1, 70], [0, 30, 60,8]],
    z: prepro4Plotly( matrixREVec ),
    type: 'heatmap',
  },
  
];


// plotting the data :
Plotly.newPlot('leftEigenVector',lVecData,{title: "Left-SingularVectors",

        font : {
            size : 15,
            color: 'white',
            family : 'Helvetica'
        },
        paper_bgcolor : '#222633',
});
Plotly.newPlot('singularValues',sVecData,{title: "SingularValues",

        font : {
            size : 15,
            color: 'white',
            family : 'Helvetica'
        },
        paper_bgcolor : '#222633',

});
Plotly.newPlot('rightEigenVector',rVecData,{title: "Right-SingularVectors",

        font : {
            size : 15,
            color: 'white',
            family : 'Helvetica'
        },
        paper_bgcolor : '#222633',
});



/* Reconstructing the image using cummulative SVD components */
let compArray =     [tf.zeros([imageDim.width, imageDim.height]), tf.matMul( tf.mul(tfU.slice([0,0],[-1,1]), tfS.slice([0,0],[1,1]) ) , tfV.slice([0,0],[1,-1]) )];
let cummCompArray = [tf.zeros([imageDim.width, imageDim.height]), tf.matMul( tf.mul(tfU.slice([0,0],[-1,1]), tfS.slice([0,0],[1,1]) ) , tfV.slice([0,0],[1,-1]) )];



// pre-calculate all the components and cummulative component images.
for(let i=2;i<tfS.shape[0]+1;i++){
  const cComp = tf.matMul( 
                            tf.mul(tfU.slice([0,i-1],[-1,1]), tfS.slice([i-1,i-1],[1,1]) ) , 
                            tfV.slice([i-1,0],[1,-1]) 
                          );
  const cummulativeComp = tf.add(cummCompArray[i-1], cComp);

  cummCompArray.push(cummulativeComp);
  compArray.push(cComp);
}

const originalImg = tf.tensor(matrix);

const slider = document.getElementById('compNum');
slider.oninput = ()=>{


  //get  current component number from user.
  let compNum = Math.floor( slider.value * tfS.shape[0] );
  document.getElementById('currCompNum').innerHTML = (compNum)? compNum: "None";

  // fetching the current component image and cummulative image upto 'compNum'
  const currComp = compArray[compNum];
  const currCummComp = cummCompArray[compNum];

  // Calcuating percent Difference:
  const pDiff = 100* (( tf.norm(currCummComp).arraySync() / tf.norm(originalImg).arraySync() ));
  document.getElementById('pctDiff').innerHTML = `  ${pDiff.toFixed(2)}%`;

  // Visualizing currently added Component
  const compData = [
    {
      z: prepro4Plotly( currComp.arraySync() ),
      // colorscale: 'Greys',
      type: 'heatmap',
      colorscale : [
                    [ 0, darkModeCols.darkBlue()],
                    [ .25, darkModeCols.magenta()], 
                    [ 0.5, darkModeCols.blue()], 
                    [ 0.75, darkModeCols.yellow()],
                    [ 1, darkModeCols.red()], 
                  
                  ],
        font : {
            size : 15,
            color: 'white',
            family : 'Helvetica'
        },
        paper_bgcolor : '#222633',
    }
  ];

  // Visualizing the Constructed Commulative Component Image:-
  const cummCompData = [
    {
      z: prepro4Plotly( currCummComp.arraySync()),
      colorscale: 'Greys',
      type:'heatmap',
    }
  ];



let cNum = compNum-1;

const eigenValsPlot = [

  // visualizing unused singular values.
  {
    y : eigenVals,
    x : tf.linspace(1, eigenVals.length+1, eigenVals.length).flatten().arraySync(),
    mode: 'lines+markers',
    type:'scatter',
    line: {color: darkModeCols.purple(),width:1} ,
    marker: {size:1},

    name: 'unused singular values(SVs)'
  },
];

if (compNum){
  eigenValsPlot.push(
  // showing the cummulative singular values used in reconstructed image.
    {
    // current component
    y : eigenVals.slice(0, cNum+1),
    x : tf.linspace(1, eigenVals.length+1, eigenVals.length).flatten().arraySync().slice(0, cNum+1),
    mode: 'lines+markers',
    type:'scatter',
    line: {color: darkModeCols.blue(),width:2} ,
    markers: {width:25},
    name: 'cummulative-SVs'
  },
  );

  // showing the currently added singular value.
  eigenValsPlot.push(

  {

    // current component
    y : [eigenVals[cNum]],
    x :[cNum+1],
    mode: 'markers',
    type:'scatter',
    marker: {color: "magenta",size:20},
    name: 'current-SV'
  },
  )


}




  // updating  plotly
Plotly.react('compViz0',
              compData,
              {
                title: {
                      text: "currently added Component: " +((compNum)? compNum: "None"),
                      font:{
                        color: "magenta",

                      }
                },
            
        font : {
            size : 15,
            color: 'white',
            family : 'Helvetica'
        },
        paper_bgcolor : '#222633',
            
            },
              {staticPlot: true}
            );
Plotly.react('cummCompViz',
              cummCompData,
              {
                // title: "Cummulative Component Image", 
                title: {
                      text:(compNum)? "Reconstructed Image \n (using components 1 to "+compNum+ ")": "Reconstructed Image using Nothing "  , 
                      font:{
                        color: darkModeCols.blue()

                      }
                },

        font : {
            size : 15,
            color: 'white',
            family : 'Helvetica'
        },
        paper_bgcolor : '#222633',
              },
              {staticPlot: true}
            );
  


Plotly.react('screePlot',eigenValsPlot,{title: "<br>Singular-Values", width: 600, height: 400,

        font : {
            size : 15,
            color: 'white',
            family : 'Helvetica',
            
        },
        plot_bgcolor: "#222633",
        paper_bgcolor : '#222633',

}, 

              {staticPlot: true}

);

};


}



