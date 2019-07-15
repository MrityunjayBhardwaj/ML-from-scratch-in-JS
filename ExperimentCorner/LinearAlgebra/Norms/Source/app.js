
/* color schemes taken from :- https://codepen.io/ruchern/pen/OgJqvr */
let darkModeCols = {
	red:   (alpha = 1)=> `rgba(255, 99, 132,${alpha})`,
	orange:(alpha = 1)=> `rgba(255, 159, 64,${alpha})`,
	yellow:(alpha = 1)=> `rgba(255, 205, 86,${alpha})`,
	green: (alpha = 1)=> `rgba(75, 192, 192,${alpha})`,
	blue:  (alpha = 1)=> `rgba(54, 162, 235,${alpha})`,
	purple:(alpha = 1)=> `rgba(153, 102, 255,${alpha})`,
    grey:  (alpha = 1)=> `rgba(231,233,237,${alpha})`,
    magenta: (alpha = 1) =>`rgba(255,0,255, ${alpha})`,
    violet: (alpha = 1) =>`rgba(255,0,255, ${alpha})`
};


const x = tf.tensor([[1],[2]]);

// generate meshgrid
const grid = meshGridRange(range={x: {min: -1.1, max: +1.1},y: {min: -1.1, max: +1.1}},division=50);

// calculate pNorm for each value of meshgrid

const  g = pNorm(tf.tensor(tf.tensor(grid[0]).transpose().arraySync()[0]).expandDims(1) )

const pNormGrid = grid.map( (a) =>{
    const f = tf.tensor(a).transpose().arraySync();
    const w = f.map( (b) =>{ 
        return 1*(inducedMatrixNorm(tf.tensor([[1, 2],[0, 2]]),tf.tensor(b).expandDims(1),p=2).flatten().arraySync()[0] )
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
    type: 'contour',
    // colorscale: 'heatmap',
    // colorscale: [[0, 'rgb(166,206,227)'], [0.25, 'rgb(31,120,180)'], [0.45, 'rgb(178,223,138)'], [0.65, 'rgb(51,160,44)'], [0.85, 'rgb(251,154,153)'], [1, 'rgb(227,26,28)']],

    colorscale : [[0, darkModeCols.blue()], [0.25, darkModeCols.purple()],[0.5, darkModeCols.magenta()], [.75, darkModeCols.yellow()], [1, darkModeCols.red()]],

    // mode: 'heatmap'
    contours: {
        // coloring: 'heatmap',
     },
    line : {
        // width: 0
    }
}];

Plotly.newPlot('pNormViz',pNormVizData,{title: 'P-Norm'});



