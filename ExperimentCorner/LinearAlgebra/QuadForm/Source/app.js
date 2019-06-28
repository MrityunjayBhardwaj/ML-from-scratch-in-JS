// TODO: make it async
const matrix = tf.tensor( [[ -1,0],[0,0]] );

function vizQuadSurf(matrix){

    // calculating quadratic form
    const matrixQuadForm = quadForm(matrix,range={min: 0, max: 1},isNormalized=1,res=10); 

    // visualizing QuadForm:-
    const quadSurfVizData = [
        { 
            x:matrixQuadForm.x,
            y:matrixQuadForm.y,
            z:matrixQuadForm.z,

            type: 'surface'

        }
    ]

    // plotting to the canvas
    Plotly.newPlot('quadSurfViz',quadSurfVizData,{title: 'Quadratic Suface'});

}

// gathring all the sliders
const slider00 = document.getElementById("range00");
const slider01 = document.getElementById("range01");
const slider10 = document.getElementById("range10");
const slider11 = document.getElementById("range11");


// creating a matrix:-
function onSliderChange(){
    const mtx00 = slider00.value*1;
    const mtx01 = slider01.value*1;
    const mtx10 = slider10.value*1;
    const mtx11 = slider11.value*1;

    // upating the current matrix 
    currMtx = [[mtx00,mtx01],[mtx10,mtx11]];
    // console.log(currMtx)
    const tfMtx = tf.tensor(currMtx);
    // updating quad surf viz
    vizQuadSurf(tfMtx);
}