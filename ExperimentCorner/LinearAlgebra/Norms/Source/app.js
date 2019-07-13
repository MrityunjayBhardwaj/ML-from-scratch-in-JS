
const x = tf.tensor([[1],[2]]);

// generate meshgrid
const grid = meshGridRange(range={x: {min: -1.1, max: +1.1},y: {min: -1.1, max: +1.1}},division=50);

// calculate pNorm for each value of meshgrid

const  g = pNorm(tf.tensor(tf.tensor(grid[0]).transpose().arraySync()[0]).expandDims(1) )

const pNormGrid = grid.map( (a) =>{
    const f = tf.tensor(a).transpose().arraySync();
    const w = f.map( (b) =>{ return 1*(pNorm(tf.tensor(b),p=1).flatten().arraySync()[0])});
    // console.log(w);
    return w;
});
console.log('grid ',grid);
console.log('pNormGrid ',pNormGrid);

// visualzing the pNorm Grid
const pNormVizData = [{
    x : grid[0][0],
    y : grid[0][0],
    z : pNormGrid,
    type: 'heatmap'
}];

Plotly.newPlot('pNormViz',pNormVizData,{title: 'P-Norm'})