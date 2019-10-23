function fn(x, weights= [10, -2], bias = 1){
    const w = tf.tensor(weights).expandDims(1);
    const b = bias;


    // doing matMul with the given weights
    return tf.clipByValue( tf.round(tf.matMul(w , x, 1, 1).add(b) ) , -1, 1 );
    // return tf.matMul(w , x, 1, 1).add(b) ;
}



const range = [-5, 5]; 
const division = 200;
const inp = tf.linspace(range[0], range[1], division).flatten().arraySync();

// const boundry = inp.map((x) => fn(x, 1));


// TODO: implement this technique to all the algos in order to get the performance boost for free
const inpMeshGridTensor = tf.tensor(meshGrid(inp, inp));


// params of hyperplane:-
let angle = (Math.PI/180)*40;
let weights = [ Math.cos(angle)*1, Math.sin(angle)*1];
let bias  = 1;

//  evalViz: 


// calculate fn on the reshaped tensor (for faster calculation ) and then reshape it back
// const outputTensor = fn(inpMeshGridTensor.reshape([inp.length**2, 2]), weights, bias).reshape([inp.length, inp.length]);
// const output = outputTensor.arraySync(); 
const output = [];

// old and sloooooowwwww way:-
// const output = inpMeshGrid.map(fn);
// const output = inpMeshGrid.map( (col) => col.map( (cell) => tf.clipByValue( Math.round(fn(cell)), -1, 1 ).flatten().arraySync()[0] ) );


const normalizedWeights = (tf.tensor(weights).div(tf.norm(weights)).transpose()).arraySync();

// point that are on the hyperplane

const x0 = [-5, 5];
const x1 = [-(weights[0]*x0[0] + bias)/weights[1], -(weights[0]*x0[1] + bias)/weights[1]];

const p0 = [x0[0], x1[0]];
const p1 = [x0[1], x1[1]];


// point that are away from hyperplane :-

function proj(x, n ){

    const u = tf.tensor(x).expandDims(1);
    n = tf.tensor(n).expandDims(1);

    return u.sub( u.transpose().matMul(n).div(tf.norm(n).pow(2)).mul(n)).flatten().arraySync()

}

const rndPt = [2, 1];
( -( bias/( (tf.norm(tf.tensor(weights))).flatten().arraySync()[0] ) ) )

const projPlane = [ normalizedWeights[1] ,
                    normalizedWeights[0] 

]

let projRndPt = proj(rndPt, projPlane)
// let projRndPt = proj(rndPt, [normalizedWeights[1], normalizedWeights[0] ]  )
// projRndPt = proj(rndPt, [normalizedWeights[1]* ( -( bias/( (tf.norm(tf.tensor(weights))).flatten().arraySync()[0] ) ) ), normalizedWeights[0]* ( -( bias/( (tf.norm(tf.tensor(weights))).flatten().arraySync()[0] ) ) ) ]  )

projRndPt[0] = projRndPt[0] + normalizedWeights[1]*( -( bias/( (tf.norm(tf.tensor(weights))).flatten().arraySync()[0] ) ) );
projRndPt[1] = projRndPt[1] + normalizedWeights[0]*( -( bias/( (tf.norm(tf.tensor(weights))).flatten().arraySync()[0] ) ) );

if( 2 >= 4)

const layout = {
    title: "Experiment",
    autosize: false,
    width: 800,
    height: 800,

    xaxis:{
        range : range,
        autorange: false
    },

    yaxis:{
        range : range,
        autorange: false
    }

}


let decisionRegionData = [ 
    // {
    //     x : inp,
    //     y : inp,
    //     z : output,
    //     type: 'contour',
    //         colorscale:[ [0, 'rgb(153, 153, 255)'], [0.5, 'rgb(255, 255, 255)'], [1, 'rgb(255, 0 , 106)']],
    //     contours : {
    //     // coloring : 'heatmap',
    //     // zsmooth: 'best',
    //     },
    //     line : {
    //     width: 0,
    //         // smoothing: 0.85
    //     },

    // },

    // -----------------------------
    // visualizing axis for reference:-
    {
        x : range,
        y: [0,0],
        type: 'scatter',

        line : {
            color : 'green'
        }
    },

    {
        x: [0,0],
        y : range,
        type: 'scatter',

        line : {
            color : 'red'
        }

    },

    // -----------------------------



    // visualizing Hyperplane:-
    {
        x : [0, normalizedWeights[1]*( -( bias/( (tf.norm(tf.tensor(weights))).flatten().arraySync()[0] ) ) )],
        y : [0, normalizedWeights[0]*( -( bias/( (tf.norm(tf.tensor(weights))).flatten().arraySync()[0] ) ) )],
        type: 'scatter',

        line: {
            color: 'black',
            width: 5,
        }
    },

    {
        // NOTE: here, we have switched the axis because the hyperplane is defined by the transpose of the weights :D
        // thats why in order to align it perfectly we need to transpose it as well which is just the interchange of the axis 
        x: [p0[1],p1[1]],
        y: [p0[0],p1[0]],
        type: 'scatter',

        line: {
            color: 'blue',
            width: 5,
        }
    },

    // randPt projection :=
    {
        x : [rndPt[0]],
        y : [rndPt[1]],

        marker :{
            size: 20
        }
    },
    

    // randPt proj on the hyperplane
    { 
        x: [projRndPt[0]],
        y: [projRndPt[1]],

        marker :{
            size: 30
        }

    },

    // visualzing the distance from x_proj to x
    {
        x : [projRndPt[0], rndPt[0]],
        y : [projRndPt[1], rndPt[1]],

        type: 'scatter',
        marker : {
            size: 1   
        },
        line : {
            dash: 'dash',
            width: 3,
            
        }

    }
];


Plotly.newPlot('expViz', decisionRegionData, layout);





// Controls:-

// hyperplane controls:-
const usrAngle = document.getElementById("angleSlider");
const usrBias = document.getElementById("biasSlider");


// random point controls : -

const usrXVal = document.getElementById("xValSlider");
const usrYVal = document.getElementById("yValSlider");

usrAngle.addEventListener("input", test);
// usrBias.addEventListener("input", onSlideUpdate);

// usrXVal.addEventListener("input", onSlideUpdate);
// usrYVal.addEventListener("input", onSlideUpdate);

let currPromise = null;
function onSlideUpdate(value){
        console.log(currPromise);

    if(currPromise === null){
    //    updateChartPromise().then((val) => {console.log(val); currPromise=null;}) ;
       test().then((val) => {console.log(val); currPromise=null;}) ;
    }

}



function updateChartPromise() {

    currPromise = onSlide();
    //     // setTimeout(() => resolve('awesome!'), 5000)
    //     onSlide(resolve);
    // });
    return currPromise;

}







console.log(decisionRegionData.length)
function test() {

    if (currPromise !== null)return null;

    currPromise = 1;
     return new Promise((resolve, reject) =>{
        // window.alert('its working')
        const degrees = usrAngle.value*360

        angle = (Math.PI/180)*degrees;
        weights = [ Math.cos(angle)*1, Math.sin(angle)*1];
        bias  = usrBias.value*range[0];


        console.log(angle);

    // calculate fn on the reshaped tensor (for faster calculation ) and then reshape it back

        console.log('started calculating the outputTensor');
            let outputTensor = [];
                    outputTensor = fn(inpMeshGridTensor.reshape([inp.length**2, 2]), weights, bias)
                    .reshape([inp.length, inp.length]);

            resolve(outputTensor);

    }).then(
        (val) =>{
            console.log('outputTensor Calculated',val);

            currPromise = null; 
        }
    )

}



















function onSlide(){

    return new Promise((resolve, reject) =>{

    // window.alert('its working')
    const degrees = usrAngle.value*360

    angle = (Math.PI/180)*degrees;
    weights = [ Math.cos(angle)*1, Math.sin(angle)*1];
    bias  = usrBias.value*range[0];


    console.log(angle);

    

    // return null;




// calculate fn on the reshaped tensor (for faster calculation ) and then reshape it back
let outputTensor = [];





for(let i=0;i< 100;i++){
    outputTensor = fn(inpMeshGridTensor.reshape([inp.length**2, 2]), weights, bias).reshape([inp.length, inp.length]);
}


const output = outputTensor.arraySync(); 
// const output = [];

// old and sloooooowwwww way:-
// const output = inpMeshGrid.map(fn);
// const output = inpMeshGrid.map( (col) => col.map( (cell) => tf.clipByValue( Math.round(fn(cell)), -1, 1 ).flatten().arraySync()[0] ) );


const normalizedWeights = (tf.tensor(weights).div(tf.norm(weights)).transpose()).arraySync();

// point that are on the hyperplane

const py = range; // points that are on the edge of our allowed input range
const px = [-(weights[0]*x0[0] + bias)/weights[1], -(weights[0]*x0[1] + bias)/weights[1]];

const p0 = [py[0], px[0]];
const p1 = [py[1], px[1]];


// point that are away from hyperplane :-

const rndPt = [-range[0]*(1-usrXVal.value) + usrXVal.value*(range[0]**1), -range[1]*(1-usrYVal.value) +  usrYVal.value*(range[1]**1)];



( -( bias/( (tf.norm(tf.tensor(weights))).flatten().arraySync()[0] ) ) )

const projPlane = [ normalizedWeights[1] ,
                    normalizedWeights[0] 

]

let projRndPt = proj(rndPt, projPlane)
// let projRndPt = proj(rndPt, [normalizedWeights[1], normalizedWeights[0] ]  )
// projRndPt = proj(rndPt, [normalizedWeights[1]* ( -( bias/( (tf.norm(tf.tensor(weights))).flatten().arraySync()[0] ) ) ), normalizedWeights[0]* ( -( bias/( (tf.norm(tf.tensor(weights))).flatten().arraySync()[0] ) ) ) ]  )

projRndPt[0] = projRndPt[0] + normalizedWeights[1]*( -( bias/( (tf.norm(tf.tensor(weights))).flatten().arraySync()[0] ) ) );
projRndPt[1] = projRndPt[1] + normalizedWeights[0]*( -( bias/( (tf.norm(tf.tensor(weights))).flatten().arraySync()[0] ) ) );



decisionRegionData = [ 
    {
        x : inp,
        y : inp,
        z : output,
        type: 'contour',
            colorscale:[ [0, 'rgb(153, 153, 255)'], [0.5, 'rgb(255, 255, 255)'], [1, 'rgb(255, 0 , 106)']],
        contours : {
        // coloring : 'heatmap',
        // zsmooth: 'best',
        },
        line : {
        width: 0,
            // smoothing: 0.85
        },

    },

    // -----------------------------
    // visualizing axis for reference:-
    {
        x : range,
        y: [0,0],
        type: 'scatter',

        line : {
            color : 'green'
        }
    },

    {
        x: [0,0],
        y : range,
        type: 'scatter',

        line : {
            color : 'red'
        }

    },

    // -----------------------------


    {
        x : [0, normalizedWeights[1]*( -( bias/( (tf.norm(tf.tensor(weights))).flatten().arraySync()[0] ) ) )],
        y : [0, normalizedWeights[0]*( -( bias/( (tf.norm(tf.tensor(weights))).flatten().arraySync()[0] ) ) )],
        type: 'scatter',

        line: {
            color: 'black',
            width: 5,
        }
    },

    {
        // NOTE: here, we have switched the axis because the hyperplane is defined by the transpose of the weights :D
        // thats why in order to align it perfectly we need to transpose it as well which is just the interchange of the axis 
        x: [p0[1],p1[1]],
        y: [p0[0],p1[0]],
        type: 'scatter',

        line: {
            color: 'blue',
            width: 5,
        }
    },

    // randPt projection :=
    {
        x : [rndPt[0]],
        y : [rndPt[1]],

        marker :{
            size: 20
        }
    },
    
    { 
        x: [projRndPt[0]],
        y: [projRndPt[1]],

        marker :{
            size: 30
        }

    },

    // visualzing the distance from x_proj to x
    {
        x : [projRndPt[0], rndPt[0]],
        y : [projRndPt[1], rndPt[1]],

        type: 'scatter',
        marker : {
            size: 1   
        },
        line : {
            dash: 'dash',
            width: 3,
            
        }

    }
];


// Plotly.animate('expViz', {data : [{x: [0,0 ]}, {y: [0, 0]}], traces: [0], layout: layout},
// {
//     transition: {duration: 500, easing: 'cubic-in-out'},
//     frame: {duration: 1000}
// })

Plotly.react('expViz', decisionRegionData,layout)
   .then(() => {console.log('its over'); setTimeout(() =>resolve('its awesome'), 10) });


    })
}

