

const vector   = tf.tensor([[1],[1]]);
const projBase = tf.tensor([[2],[0.5]]);
const projVec  = project(projBase, vector);
let orthoVec = tf.sub(projVec, vector);

orthoVec = projVec.add(tf.neg(orthoVec));

const vecVizData = [
    {
        x : [0,vector.flatten().arraySync()[0]],
        y : [0,vector.flatten().arraySync()[1]],
        mode: 'lines',
        type : 'scatter',
    },
    {
        x : [0,projBase.flatten().arraySync()[0]],
        y : [0,projBase.flatten().arraySync()[1]],
        mode: 'lines',
        type : 'scatter',
    },
    {
        x : [projVec.flatten().arraySync()[0]],
        y : [projVec.flatten().arraySync()[1]],
        // mode: 'lines',
        type : 'scatter',
    },
    {
        x : [projVec.flatten().arraySync()[0],orthoVec.flatten().arraySync()[0]],
        y : [projVec.flatten().arraySync()[1],orthoVec.flatten().arraySync()[1]],
        mode: 'lines',
        type : 'scatter',
    }

]
Plotly.newPlot('vecViz',vecVizData)