

const x = tf.tensor([[1],[2]]);

// generate meshgrid
const grid = meshGridRange(range={x: {min: -1.1, max: +1.1},y: {min: -1.1, max: +1.1}},division=50);

// calculate pNorm for each value of meshgrid

const  g = pNorm(tf.tensor(tf.tensor(grid[0]).transpose().arraySync()[0]).expandDims(1) )

const pNormGrid = grid.map( (a) =>{
    const f = tf.tensor(a).transpose().arraySync();
    const w = f.map( (b) =>{ 
        return 1*(pNorm(tf.tensor(b).expandDims(1), p=2).flatten().arraySync()[0] )
        // return 1*(inducedMatrixNorm(tf.tensor([[1, 2],[0, 2]]),tf.tensor(b).expandDims(1),p=2).flatten().arraySync()[0] )
    });
    return w;
});

console.log('grid ',grid);
console.log('pNormGrid ',pNormGrid);
    
// visualzing the pNorm Grid
const pNormVizData = [{
    x : grid[0][0],
    y : grid[0][0],
    z : pNormGrid,
    type: 'surface',

    colorscale : [[0, darkModeCols.blue()], [0.25, darkModeCols.purple()],[0.5, darkModeCols.magenta()], [.75, darkModeCols.yellow()], [1, darkModeCols.red()]],

    contours: {
     },
    line : {
    },
}];

const layoutSetting = {
    title : 'p-Norm',
    font : {
        size : 15,
        color: 'white',
        family : 'Helvetica'
    },
    paper_bgcolor : '#222633',


}

Plotly.newPlot('pNormViz',pNormVizData,layoutSetting);



