let mIrisX = tf.tensor(iris).slice([0,1],[-1,2]);
// one hot encoded
// let mIrisY = tf.tensor( Array(100).fill([1,0],0,50).fill([0,1],50) );

let mIrisY = tf.tensor( Array(150).fill([1,0,0], 0, 50).fill([0,1,0], 50, 100).fill([0,0,1], 100) );
const standardDataX = normalizeData(mIrisX, 1)

const {0: trainData,1: testData} = trainTestSplit(standardDataX,mIrisY,2/3);

// const classwiseDataSplit

const X = tf.linspace(-5,5, 80).expandDims(1);

noise = 0.4;
// X_train = tf.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).expandDims(1);
X_train = tf.linspace(-4,5,10).expandDims(1);
Y_train = tf.sin(X_train).add( tf.randomNormal(X_train.shape).mul(noise) );

const model = new gaussianProcess;

// X_train = np.arange(-3, 4, 1).reshape(-1, 1)
// Y_train = np.sin(X_train) + noise * np.random.randn(*X_train.shape)

const samples = model.test(X, data={x: X_train, y: Y_train});

// const samples = [[ 0.06730658,  0.75677232, -0.14110455, -0.90935443, -0.84150691,  0.53829706,  0.84135292, -1.52066521, -2.46284952, -0.81536249],
// [ 0.66076553,  0.75696553, -0.14109077, -0.90942998, -0.84158982,  0.87194648,  0.84150261,  0.96797996,  1.49691305,  1.6875946 ]]

// plotting samples


var dataGaussian = [
    {
        x: X_train.flatten().arraySync(),
        y: Y_train.flatten().arraySync(),
        mode: 'markers',
        type: 'scatter',
        
        marker: {
            size: 30,
            color: 'red'
        }
    },


    {
        x: X.flatten().arraySync(),
        y: samples[0],
        // mode: 'markers',
        type: 'scatter',
    },{
        x: X.flatten().arraySync(),
        y: samples[1],
        
        // mode: 'markers',
        type: 'scatter',
    },{
        x: X.flatten().arraySync(),
        y: samples[2],
        // mode: 'markers',
        type: 'scatter',
    },
];

var GaussianLayout = {
title: ' parameter Space',
autosize: false,
width: 800,
height: 800,
};

Plotly.newPlot('parameterSpace',dataGaussian,GaussianLayout);

