function normal() {
    var x = 0,
        y = 0,
        rds, c;
    do {
        x = Math.random() * 2 - 1;
        y = Math.random() * 2 - 1;
        rds = x * x + y * y;
    } while (rds == 0 || rds > 1);
    c = Math.sqrt(-2 * Math.log(rds) / rds); // Box-Muller transform
    return x * c; // throw away extra sample y * c
}

var N = 2000,
  a = -1,
  b = 1.2;

var step = (b - a) / (N - 1);
var t = new Array(N), x = new Array(N), y = new Array(N);

for(var i = 0; i < N; i++){
  t[i] = a + step * i;
  x[i] = (Math.pow(t[i], 1))*1 + (.8 * normal() );
  y[i] = (Math.pow(t[i], 1))*1 + (.3 * normal() );

  x[i] = [x[i],y[i]];
}


let data0 = tf.randomNormal([1,500],0,1).transpose();
let data1 = tf.randomNormal([1,500],0,1).transpose();

let data = data0.concat(data1,axis=1).arraySync();

const mIrisX = tf.tensor(iris).slice([0,0],[100,2])

// one hot encoded
const mIrisY = Array(100).fill([1,0],0,50).fill([0,1],50);

const model = new FDA;

const weights = model.train({x:mIrisX.arraySync(),y:mIrisY});

const maxDataVals = mIrisX.max(0).arraySync();
const minDataVals = mIrisX.min(0).arraySync();

const tSize = 50;

// Creating Simulated Data X
const simX0 = tf.linspace(minDataVals[0],maxDataVals[0],tSize).expandDims(1);
const simX1 = tf.linspace(minDataVals[1],maxDataVals[1],tSize).expandDims(1);

const simX  = simX0.concat(simX1,axis=1);

let sY = tf.matMul(mIrisX,weights);

// sY = tf.matMul(weights,sY);
// const simY = model.test(mIrisX.slice([0,0],[50,-1]).arraySync());

const projX0 = [];
const projX1 = [];

for(let i=0;i<sY.shape[0];i++){

  const currVal = sY.slice([i],[1]);

  const projData = tf.matMul(weights,currVal).flatten().arraySync();

  projX0.push(projData[0]);
  projX1.push(projData[1]);

}

// normalized Data X
const normX = normalize(mIrisX,1)

// sY = sY.flatten().arraySync();

const p = 50;

const fac = 10;
// Visualizing mIrisX
const dataXViz = [{
  x: mIrisX.slice([0,0],[p,-1]).slice([0,0],[-1,1]).reshape([p,]).arraySync(),
  y: mIrisX.slice([0,0],[p,-1]).slice([0,1],[-1,1]).reshape([p,]).arraySync(),
  mode: 'markers',
  type: 'scatter',
  markers : {width : 4}
},
{
  x: mIrisX.slice([p,0],[-1,-1]).slice([0,0],[-1,1]).reshape([p,]).arraySync(),
  y: mIrisX.slice([p,0],[-1,-1]).slice([0,1],[-1,1]).reshape([p,]).arraySync(),
  mode: 'markers',
  type: 'scatter',
  markers : {width : 4}
},
// plotting FDA transformed x values 
{
  // y: Array(50).fill(0) ,
  // x: sY.slice(0,50),
  x: projX0.slice(0,p),
  y: projX1.slice(0,p),
  mode: 'markers',
  type: 'scatter',
  markers: {width: 4}
},
{
  // y: Array(50).fill(0) ,
  // x: sY.slice(50,),
  x: projX0.slice(p,),
  y: projX1.slice(p,),
  mode: 'markers',
  type: 'scatter',
  markers: {width: 4}
},
// Plotting FDA projection vector
{
  x: [-weights.flatten().arraySync()[0]*fac, weights.flatten().arraySync()[0]*fac ],
  y: [-weights.flatten().arraySync()[1]*fac, weights.flatten().arraySync()[1]*fac ],
  mode: 'lines',
  type: 'scatter',
  lines: {width: 4}
}
]
Plotly.newPlot('inputSpace',dataXViz,{title: 'inputSpace'})

// testing psudo inverse:-
const A =  [ [-0.63992,  0.56163,  0.66743],
  [ 1.25050, -1.27542, -1.45304],
  [ 0.41833, -0.58368, -0.47264]];

pinv(A);

// plotting projected output
const projOutDta = [
{
  y: Array(50).fill(0) ,
  x: sY.flatten().arraySync().slice(0,50),
  mode: 'markers',
  type: 'scatter',
  markers: {width: 4}
},
{
  y: Array(50).fill(0) ,
  x: sY.flatten().arraySync().slice(50,),
  mode: 'markers',
  type: 'scatter',
  markers: {width: 4}
},
];

Plotly.newPlot('projOut',projOutDta,{title: 'FDA projected X'})

// 3d plotting

// const for3d = mIrisX.slice([0,0],[-1,1]).concat(sY,axis=1).concat(mIrisX.slice([0,-1],[-1,1]),axis=1).transpose().arraySync();
// const for3d = mIrisX.concat(sY,axis=1).arraySync();







// Experiment | ---------------------











let  modX = tf.tensor(iris).slice([0,0],[p*2,3]).arraySync();
// modX = mIrisX.arraySync() ;

const multiIrisY = Array(150).fill([1,0,0],0,50).fill([0,1,0],50,100).fill([0,0,1],100);

const multmodel = new FDAmc();
const wts = multmodel.train({x:  modX , y: mIrisY /* 2 classes */ });
const modWts = tf.tensor(wts).slice([0,1],[-1,2]);
let tfModX = tf.tensor(modX);
let pridX = tf.matMul( tfModX, modWts );

// preprocess for plotting:-
pridX = pridX.concat(tf.zeros([pridX.shape[0],3-pridX.shape[1]]),axis=1);
tfModX = tfModX.concat(tf.zeros([tfModX.shape[0],3-tfModX.shape[1]]),axis=1);

const qrModWts = tf.linalg.qr(modWts)[0];
let basis0 = qrModWts.slice([0,0], [-1,1]).flatten().arraySync();
let basis1 = qrModWts.slice([0,1], [-1,1]).flatten().arraySync();

const spanLen = 5;

const spanMesh = [];

const vec0 = basis0.map( (a)=> a*spanLen);
const vec1 = basis1.map( (a)=> a*spanLen);

// spanMesh.push()
let meshPlane = [
                [-vec0[0],vec1[0],vec0[0],-vec1[0]],
                [-vec0[1],vec1[1],vec0[1],-vec1[1]],
                [-vec0[2],vec1[2],vec0[2],-vec1[2]],
              ];

basis0 = modWts.slice([0,0], [-1,1]).flatten().arraySync();
basis1 = modWts.slice([0,1], [-1,1]).flatten().arraySync();
// meshPlane  =[
//    [0, 0, 1, 1, ],
//    [0, 1, 1, 0, ],
//    [0, 0, 0, 0, ],
// ];

  var dataFDA = [{
            x: pridX.slice([0,0],[p,-1]).slice([0,0],[-1,1]).flatten().arraySync() || Array(tfModX.shape[0]).fill(0),
            y: pridX.slice([0,0],[p,-1]).slice([0,1],[-1,1]).flatten().arraySync() || Array(tfModX.shape[0]).fill(0),
            mode: 'markers',
            type: 'scatter',
         },
        {
            x: pridX.slice([p,0],[-1,-1]).slice([0,0],[-1,1]).flatten().arraySync() || Array(tfModX.shape[0]).fill(0),
            y: pridX.slice([p,0],[-1,-1]).slice([0,1],[-1,1]).flatten().arraySync() || Array(tfModX.shape[0]).fill(0),
            mode: 'markers',
            type: 'scatter',
          },
//      {
//             x: pridX.slice([p*2,0],[-1,-1]).slice([0,0],[-1,1]).flatten().arraySync() || Array(tfModX.shape[0]).fill(0),
//             y: pridX.slice([p*2,0],[-1,-1]).slice([0,1],[-1,1]).flatten().arraySync() || Array(tfModX.shape[0]).fill(0),
//             z: pridX.slice([p*2,0],[-1,-1]).slice([0,2],[-1,1]).flatten().arraySync() || Array(tfModX.shape[0]).fill(0),
//             mode: 'markers',
//             type: 'scatter3d',
//           }
        ];
    
  var FDALayout = {
    title: ' FDA transformed data',
    autosize: false,
    width: 500,
    height: 500,
  };

  Plotly.newPlot('3dPlot',dataFDA,FDALayout);

  var dataOriginal = [
        {
            x: tfModX.slice([0,0],[p,-1]).slice([0,0],[-1,1]).flatten().arraySync() || Array(tfModX.shape[0]).fill(0),
            y: tfModX.slice([0,0],[p,-1]).slice([0,1],[-1,1]).flatten().arraySync() || Array(tfModX.shape[0]).fill(0),
            z: tfModX.slice([0,0],[p,-1]).slice([0,2],[-1,1]).flatten().arraySync() || Array(tfModX.shape[0]).fill(0),
            mode: 'markers',
            type: 'scatter3d',
         },
        // {
        //     x: tfModX.slice([p,0],[100,-1]).slice([0,0],[-1,1]).flatten().arraySync() || Array(tfModX.shape[0]).fill(0),
        //     y: tfModX.slice([p,0],[100,-1]).slice([0,1],[-1,1]).flatten().arraySync() || Array(tfModX.shape[0]).fill(0),
        //     z: tfModX.slice([p,0],[100,-1]).slice([0,2],[-1,1]).flatten().arraySync() || Array(tfModX.shape[0]).fill(0),
        //     mode: 'markers',
        //     type: 'scatter3d',
        //   },
        {
            x: tfModX.slice([p*1,0],[-1,-1]).slice([0,0],[-1,1]).flatten().arraySync() || Array(tfModX.shape[0]).fill(0),
            y: tfModX.slice([p*1,0],[-1,-1]).slice([0,1],[-1,1]).flatten().arraySync() || Array(tfModX.shape[0]).fill(0),
            z: tfModX.slice([p*1,0],[-1,-1]).slice([0,2],[-1,1]).flatten().arraySync() || Array(tfModX.shape[0]).fill(0),
            mode: 'markers',
            type: 'scatter3d',
        },
        // plotting Basis of this projected X subspace:-
{
          x: [-vec0[0]*0,basis0[0]],
          y: [-vec0[1]*0,basis0[1]],
          z: [-vec0[2]*0,basis0[2]],
          mode: 'lines',
          type: 'scatter3d',
          opacity : 1,
          line: {
            width: 5 
          }


        },
{
          x: [-vec1[0]*0,basis1[0]],
          y: [-vec1[1]*0,basis1[1]],
          z: [-vec1[2]*0,basis1[2]],
          mode: 'lines',
          type: 'scatter3d',
          opacity : 1,
          line : {
            width: 5 
          }

        },
        {  
    type: "mesh3d",
    x: meshPlane[0],
    y: meshPlane[1],
    z: meshPlane[2],
    opacity : 0.7,
    color: 'pink'
  }
      ];

   Plotly.newPlot('originalData',dataOriginal,{title: 'Original Data X',})