
// initializing data
const mIrisX = tf.tensor(iris).slice([0,0],[100,2])
// one hot encoded
const mIrisY = Array(100).fill([1,0],0,50).fill([0,1],50);

let augIrisData = mIrisX.concat(tf.tensor(mIrisY),axis=1).arraySync();
tf.util.shuffle(augIrisData,axis=0);
augIrisData = tf.tensor( augIrisData );

const testX =   augIrisData.slice([0,0],[25,2]);
const testY =   augIrisData.slice([0,2],[25,-1]);

testX.print();
testY.print();
let model = new KNN();
model.train({ x: mIrisX, y: mIrisY },K=5);
const predY = model.classify(testX);



const inputSpaceVizData = [{
    x: mIrisX.slice([0,0],[50,-1]).slice([0,0],[-1,1]).arraySync(),
    y: mIrisX.slice([50,0],[-1,-1]).slice([0,1],[-1,-1]).arraySync(),

    mode: 'markers',
    type: 'scatter',

    marker : {
        width: 5
    }
}]

Plotly.newPlot('inputSpaceViz',inputSpaceVizData,{title: 'input Space'})