const mtx0Div = document.getElementById('mtx0');
const mtx1Div = document.getElementById('mtx1');

function projectionViz(){

    /* Rotate the basis of column space using slider */

    // get the angle from the sliders
    let angle0 = -Math.PI +  (document.getElementById('mtx0').value/100 )*2*Math.PI;
    let angle1 = -Math.PI +  (document.getElementById('mtx1').value/100 )*2*Math.PI;

    // print angles to the HTML
    document.getElementById('val0').innerHTML = angle0.toFixed(2);
    document.getElementById('val1').innerHTML = angle1.toFixed(2);

    // construct rotation matrix for both the basis
    let rotMtx0 = [ [Math.cos(angle0)*1   , - Math.sin(angle0)],[Math.sin(angle0) , Math.cos(angle0)]];
    let rotMtx1 = [ [Math.cos(angle1)*1   , - Math.sin(angle1)],[Math.sin(angle1) , Math.cos(angle1)]];

    // calculate the rotation vector from rotation Matrix
    let rotVec0 = JSON.parse( nd.la.matmul( nd.array(rotMtx0) , nd.array([1,0]).T ).toString() );
    let rotVec1 = JSON.parse( nd.la.matmul( nd.array(rotMtx1) , nd.array([0,1]).T ).toString() );

    // combine the rotated vectors to form our final matrix
    matrix = nd.array( [ [ rotVec0[0][0],rotVec1[0][0] ],[rotVec0[1][0],rotVec1[1][0]] ] );

    const projBaseMag = 2;

    const vector   = tf.tensor([rotVec0[0],rotVec0[1]]);
    const projBase = tf.tensor([rotVec1[0],rotVec1[1]]).mul(projBaseMag);
    const projVec  = project(projBase, vector);

    let orthoVec = tf.sub(projVec, vector);
    orthoVec = projVec.add(tf.neg(orthoVec));


    const vecVizData = [
        {
            name: 'original Vector',
            x : [0,vector.flatten().arraySync()[0]],
            y : [0,vector.flatten().arraySync()[1]],
            mode: 'lines',
            type : 'scatter',
        },
        {
            name : 'projection Base',
            x : [-projBase.flatten().arraySync()[0],projBase.flatten().arraySync()[0]],
            y : [-projBase.flatten().arraySync()[1],projBase.flatten().arraySync()[1]],
            mode: 'lines',
            type : 'scatter',
        },
        {
            name : 'ortho Vector',
            x : [projVec.flatten().arraySync()[0],orthoVec.flatten().arraySync()[0]],
            y : [projVec.flatten().arraySync()[1],orthoVec.flatten().arraySync()[1]],
            mode: 'lines',
            type : 'scatter',
        },

        {
            name : 'projection Vector',
            x : [projVec.flatten().arraySync()[0]],
            y : [projVec.flatten().arraySync()[1]],
            // mode: 'lines',
            type : 'scatter',
            marker : {
                size: 7 
            }
        },

    ];

    const vecVizLayout = {
        title : '2D-Projection',
        xaxis: {
            range: [-2,+2],
            fixedrange: true
        },
        yaxis: {
            range: [-2,+2],
            fixedrange: true
        }
    };

    Plotly.newPlot('vecViz', vecVizData, vecVizLayout);

    console.log("skdjf")
}

projectionViz();

// setInterval(() => {
//     projectionViz() 
// }, 100);