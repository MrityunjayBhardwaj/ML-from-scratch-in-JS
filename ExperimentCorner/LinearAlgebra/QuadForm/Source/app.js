// TODO: make it async
const matrix = tf.tensor( [[ -1,0],[0,0]] );


const qSLayout = {
    title: 'Quadratic Suface',
    uirevision:'true',
    xaxis: {autorange: true},
    yaxis: {autorange: true},
    zaxis: {autorange: true},
    "titlefont": {
    "size": 36
  },
}

function vizQuadSurf(matrix,isNormalized=1,res=50,){

    // calculating quadratic form
    const matrixQuadForm = quadForm(matrix,range={min: -1, max: 1},isNormalized,res); 

    // console.log(matrixQuadForm)
    // visualizing QuadForm:-
    const quadSurfVizData = [
        { 
            x:matrixQuadForm.x,
            y:matrixQuadForm.y,
            z:matrixQuadForm.z,

            type: 'surface',

        }
    ]

    // plotting to the canvas
    Plotly.newPlot('quadSurfViz',quadSurfVizData,qSLayout);

    const quadSurfHeatMapVizData = [
        {
            x: matrixQuadForm.x,
            y: matrixQuadForm.y,
            z: matrixQuadForm.z,
            type: 'heatmap'
        }
    ]

    Plotly.newPlot('quadSurfHeadMapViz',quadSurfHeatMapVizData,{margin: {pad:5,t:35,l:35},})
}


vizQuadSurf(tf.zeros([2,2]))


let currMtx = [[0,0],[0,0]];


// gathring all the sliders
const slider00 = document.getElementById("range00");
const slider01 = document.getElementById("range01");
const slider10 = document.getElementById("range10");
const slider11 = document.getElementById("range11");

const matrixVal00 = document.getElementById("matrixValue00");
const matrixVal01 = document.getElementById("matrixValue01");
const matrixVal10 = document.getElementById("matrixValue10");
const matrixVal11 = document.getElementById("matrixValue11");

const normToggle = document.getElementById("normalizedToggle");

let normalizeOrNot = normToggle.value;

// creating a matrix:-
function onSliderChange(){
    const mtx00 = slider00.value*1;
    const mtx01 = slider01.value*1;
    const mtx10 = slider10.value*1;
    const mtx11 = slider11.value*1;

    // update the ui number 
    matrixVal00.innerHTML = mtx00;
    matrixVal01.innerHTML = mtx01;
    matrixVal10.innerHTML = mtx10;
    matrixVal11.innerHTML = mtx11;

    normalizeOrNot = normToggle.checked;

    // upating the current matrix 
    currMtx = [[mtx00,mtx01],[mtx10,mtx11]];
    // console.log(currMtx)
    const tfMtx = tf.tensor(currMtx);

    console.log(normalizeOrNot)
    // updating quad surf viz
    vizQuadSurf(tfMtx,normalizeOrNot);
}


function onNormToggleChange(toggle){
    // console.log(this.checked)
    normalizeOrNot = normToggle.checked;
    const tfMtx = tf.tensor(currMtx);
    vizQuadSurf( tfMtx , normalizeOrNot);
}
