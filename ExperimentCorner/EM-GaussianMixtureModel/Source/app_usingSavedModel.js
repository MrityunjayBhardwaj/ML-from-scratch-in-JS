// initializing data
let mIrisX = tf.tensor(iris).slice([0,1],[150,2])
mIrisX = normalizeData(mIrisX,unitVariance=1);
// one hot encoded
let mIrisY = Array(150).fill([1,0],0,50).fill([0,1],50);

let augIrisData = mIrisX.concat(tf.tensor(mIrisY),axis=1).arraySync();
tf.util.shuffle(augIrisData,axis=0);
augIrisData = tf.tensor( augIrisData );

let testX =   augIrisData.slice([0,0],[25,2]);
let testY =   augIrisData.slice([0,2],[25,-1]);

const mIrisXArray = mIrisX.arraySync();
tf.util.shuffle(mIrisXArray);
mIrisX = tf.tensor(mIrisXArray);

const k = 10;

const myMeans = JSON.parse(meanString); 
const myCovariance = JSON.parse(covarianceString);
const myMixingCoeff = JSON.parse(mixingCoeffString);

const currTimeStep = 0;

const mean = [];
const covariance = [];
const mixingCoeff =  [];

for(let i=0;i<k;i++){

    mean.push(tf.tensor(myMeans[currTimeStep][i]))
    covariance.push(tf.tensor(myCovariance[currTimeStep][i]))
    mixingCoeff.push(tf.tensor(myMixingCoeff[currTimeStep][i]))

}
let myModel = new GMM(mean, covariance, mixingCoeff);

// const myMeans = [];
// const myCovariance = [];
// const myMixingCoeff = [];


// let myModel = new GMM();
// const { mean: mean, covariance: covariance, mixingCoeff: mixingCoeff} = myModel.train( mIrisX, k);

const gridRange0 = [-2, 2];
const gridRange1 = [-3, 3.5];
const psudoPtsGridRes = 100;

const psudoPts0 = tf.linspace( gridRange0[0],gridRange0[1],psudoPtsGridRes).expandDims(1);
const psudoPts1 = tf.linspace( gridRange1[0],gridRange1[1],psudoPtsGridRes).expandDims(1);
const psudoPts  = psudoPts0.concat(psudoPts1,axis=1);


let meshGridPsudoPts = tf.tensor(meshGrid(psudoPts1.flatten().arraySync(),psudoPts0.flatten().arraySync()))
meshGridPsudoPts = meshGridPsudoPts.reshape([meshGridPsudoPts.shape[0]*meshGridPsudoPts.shape[1], meshGridPsudoPts.shape[2]]);
let meshGridZ = myModel.test(meshGridPsudoPts)

const layoutSetting = {
    title: 'Gaussian Mixture Model',
    font : {
        size : 15,
        color: 'white',
        family : 'Helvetica'
    },
    paper_bgcolor : '#222633',
    width: 800,
    height: 800,

}


const clusterData = [

        // {
        //     x: mIrisX.slice([0,0],[-1,1]).flatten().arraySync(),
        //     y: mIrisX.slice([0,1],[-1,-1]).flatten().arraySync(),
        //     z: (new Array(mIrisX.shape[0])).fill(0),
        //     type: 'scatter',
        //     // type: 'scatter3d',

        //     mode: 'markers',
        //     name: ' data',
        //     opacity: .4,

        //     marker : {
        //         size: 20,
        //         color: 'orange'
        //     }
        // },

        {  
            x: meshGridPsudoPts.slice([0,0],[-1,1]).flatten().arraySync(),
            y: meshGridPsudoPts.slice([0,1],[-1,1]).flatten().arraySync(),
            z: meshGridZ.flatten().arraySync(),
            intensity: meshGridZ.flatten().arraySync(),
            type: "mesh3d",
            type: 'contour',
            // opacity : 0.7,
            // color: 'pink'

            colorscale: [
                [0, darkModeCols.blue()],
                [0.25, darkModeCols.purple()],
                [0.5, darkModeCols.magenta()],
                [0.75, darkModeCols.yellow()],
                [1, darkModeCols.red()]
            ],
        }



];



// for(let i=0; i<k; i++){

//     const currColor = 'rgb('+Math.random()*255+','+Math.random()*255+','+Math.random()*255+')';
//     // plotData.push(
//     //     {
//     //         x: clusterData[i].slice([0,0],[-1,1]).flatten().arraySync(),
//     //         y: clusterData[i].slice([0,1],[-1,-1]).flatten().arraySync(),

//     //         mode: 'markers',
//     //         type: 'scatter',
//     //         legendgroup: 'group'+i+'',
//     //         name: 'cluster'+i+' data',

//     //         marker : {
//     //             size: 10,
//     //             color: currColor,
//     //         }
//     //     }
//     // )

//     clusterData.push(
//         {

//             x: [mean[i].flatten().arraySync()[0]],
//             y: [mean[i].flatten().arraySync()[1]],


//             mode: 'markers',
//             type: 'scatter',
//             legendgroup: 'group'+i+'',
//             name: 'cluster'+i+' mean',

//             marker : {
//                 size: 50,
//                 color: currColor,
//                 opacity: .7,
//                 line: {
//                     color: 'rgb(231, 99, 250)',
//                     width: 6
//                   },
//             }
//         }
//     )
    
// }



Plotly.newPlot('GMMViz', clusterData, layoutSetting)



// // convert and save model
// let myConvertedMeans = [];
// let myConvertedCovariance = [];
// let myConvertedMixingCoeff = [];
// for(let i=0;i<myMeans.length;i++){
  
//   const currItrMeans = [];
//   const currItrCovariance = [];
//   const currItrMixingCoeff = [];
  
//   for(let j = 0;j< myMeans[0].length;j++){
//     currItrMeans.push( myMeans[i][j].arraySync());
//     currItrCovariance.push( myCovariance[i][j].arraySync());
//     currItrMixingCoeff.push( myMixingCoeff[i][j].flatten().arraySync()[0]);
//   }
  
//   myConvertedMeans.push(currItrMeans);
//   myConvertedCovariance.push(currItrCovariance);
//   myConvertedMixingCoeff.push(currItrMixingCoeff);
// }

// (JSON.stringify(myConvertedMeans) )