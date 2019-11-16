// initializing data
let mIrisX = tf.tensor(iris).slice([0,1],[100,2])
mIrisX = normalizeData(mIrisX,unitVariance=1);
// one hot encoded
let mIrisY = Array(100).fill([1,0],0,50).fill([0,1],50);

let augIrisData = mIrisX.concat(tf.tensor(mIrisY),axis=1).arraySync();
tf.util.shuffle(augIrisData,axis=0);
augIrisData = tf.tensor( augIrisData );

let testX =   augIrisData.slice([0,0],[25,2]);
let testY =   augIrisData.slice([0,2],[25,-1]);

const mIrisXArray = mIrisX.arraySync();
tf.util.shuffle(mIrisXArray);
mIrisX = tf.tensor(mIrisXArray);

let model = new KMeans();
const k = 5;
const { mean: mean, cluster: cluster} = model.train( mIrisX, k,);


const clusterData = [];

const plotData = [];

for(let i=0; i<k; i++){

    clusterData.push( mIrisX.mul(cluster.slice([0,i],[-1,1])));

    const currColor = 'rgb('+Math.random()*255+','+Math.random()*255+','+Math.random()*255+')';
    plotData.push(
        {
            x: clusterData[i].slice([0,0],[-1,1]).flatten().arraySync(),
            y: clusterData[i].slice([0,1],[-1,-1]).flatten().arraySync(),

            mode: 'markers',
            type: 'scatter',
            legendgroup: 'group'+i+'',
            name: 'cluster'+i+' data',

            marker : {
                size: 10,
                color: currColor,
            }
        }
    )

    plotData.push(
        {

            x: [mean[i][0]],
            y: [mean[i][1]],


            mode: 'markers',
            type: 'scatter',
            legendgroup: 'group'+i+'',
            name: 'cluster'+i+' mean',

            marker : {
                size: 50,
                color: currColor,
                opacity: .7,
                line: {
                    color: 'rgb(231, 99, 250)',
                    width: 6
                  },
            }
        }
    )
    
}


Plotly.newPlot('KMeansViz', plotData, {title: 'K-Means Clustering', height: 800, width:800})

