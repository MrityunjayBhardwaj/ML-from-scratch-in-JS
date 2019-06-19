// import { svd_decomp } from "../../dependency/ndjs/src/la";


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

      if (nFeatures === 1 ){
        jsArray[i] = arraySerialized[index][1];
      }
      else{
        cArrayRow[j] = arraySerialized[index][1] ;
      }

    }
    if(nFeatures !== 1)
    jsArray[i] = cArrayRow;
  }
  return jsArray;
}


// Singular Value Decomposition:-
let matrix = nd.array([[1,2],[3,1]]);

let dSize = [40,40];

const m = 40;
const k = Math.round(m/2);


const x = tf.linspace(-3,3,k).expandDims(1);

const xMatrix = x.transpose().tile([k,1]);

const g2d = tf.exp( tf.div( tf.mul( tf.scalar(-1) , tf.add(tf.pow( xMatrix,2) , tf.pow( xMatrix.transpose(),2) ) ) , tf.scalar(k/8) ));

// g2d.print();
const A = tf.truncatedNormal([m,m])
// const W = tf.conv2d(A,g2d.reshape([k,k,1,1]));

matrix = nd.array(A.arraySync());

// const IM = new ImageParser();
// const Im =p5.LoadImage('../Assets/Images/brain_sticker.png', ()=>{ console.log("it Works!")},()=>{ console.log("it Works!")});

// Create Conv2d;
for(let i=0;i<5;i++){
  for(let j=0;i<5;i++){
    break;
  }
} 

const plotMatrix = convert2dArray(matrix);

var data = [
  {
    // z: [[1, 20, 30, 30], [20, 1, 60, 50], [30, 60, 1, 70], [0, 30, 60,8]],
    z: plotMatrix,
    type: 'heatmap'
  },
  
];

Plotly.newPlot('tester', data,{title:'original Matrix \'A\''});
// Plotly.react()


// SVD :-
const matrixSvd = nd.la.svd_decomp(plotMatrix);

const matrixLEVec =  convert2dArray(matrixSvd[0]);
const matrixSEVal =  convert2dArray(nd.la.diag_mat(matrixSvd[1]));
const matrixREVec =  convert2dArray(matrixSvd[2]);

const eigenVals = convert2dArray(matrixSvd[1])
var lVecData = [
  {
    // z: [[1, 20, 30, 30], [20, 1, 60, 50], [30, 60, 1, 70], [0, 30, 60,8]],
    z:matrixLEVec,
    type: 'heatmap'
  },
  
];
var sVecData = [
  {
    // z: [[1, 20, 30, 30], [20, 1, 60, 50], [30, 60, 1, 70], [0, 30, 60,8]],
    z: matrixSEVal,
    type: 'heatmap'
  },
  
];

var rVecData = [
  {
    // z: [[1, 20, 30, 30], [20, 1, 60, 50], [30, 60, 1, 70], [0, 30, 60,8]],
    z: matrixREVec,
    type: 'heatmap'
  },
  
];

const eigenValsPlot = [
  {
    y : eigenVals,
    mode: 'lines+markers',
    type:'scatter',
    line: {color: "violet",width:3} ,
    markers: {width:10}

  }
];

Plotly.newPlot('leftEigenVector',lVecData,{title: "U"});
Plotly.newPlot('singularValues',sVecData,{title: "E"});
Plotly.newPlot('rightEigenVector',rVecData,{title: "V"});

Plotly.newPlot('screePlot',eigenValsPlot,{title: "EigenValues"});



/* Reconstructing the image using some SVD components */

// converting the U S V to tf.tensor for better calcuation

const tfU = tf.tensor(matrixLEVec);
const tfS = tf.tensor(matrixSEVal);
const tfV = tf.tensor(matrixREVec);

const compArray = [tf.zeros(tfS.shape)];
// pre-calculate all the components
for(let i=0;i<tfS.shape[0];i++){
  const cComp = tf.matMul( tf.mul(tfU.slice([0,i],[-1,1]), tfS.slice([i,i],[1,1]) ) , tfV.slice([i,0],[1,-1]) );
  // cummulativeComp = tf.add(cummulativeComp) 
  compArray.push(cComp);
}

const originalImg = tf.tensor(plotMatrix);

setInterval(() => {

  //get  current component number from user.
  let compNum = Math.floor( document.getElementById('compNum').value * tfS.shape[0] );
  document.getElementById('currCompNum').innerHTML = compNum;

  // calculate the cummulative image.
  let cummulativeComp = compArray[0];
  for(let i=0;i<compNum;i++){
    cummulativeComp = tf.add(cummulativeComp,compArray[i+1]);
  }

  // Calcuating percent Difference:
  const pDiff = 100* ( tf.norm(cummulativeComp).arraySync() / tf.norm(originalImg).arraySync() );
  document.getElementById('pctDiff').innerHTML = `  ${pDiff.toFixed(1)}%`;

  // console.log(pDiff, tf.norm(cummulativeComp).arraySync());

  // Visualizing currently added Component
  const currComp = compArray[compNum];
  const compData = [
    {
      z: currComp.arraySync(),
      type: 'heatmap'
    }
  ];

  // Visualizing the Constructed Commulative Component Image:-
  const cummCompData = [
    {
      z: cummulativeComp.arraySync(),
      type:'heatmap'
    }
  ];

  // updating  plotly
Plotly.react('compViz0',compData,{title: "Component " +compNum})
Plotly.react('cummCompViz',cummCompData,{title: "Cummulative Component Image"})

  
}, 250);





