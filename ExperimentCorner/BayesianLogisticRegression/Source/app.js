// initializing data
const IrisX = tf.tensor(iris).slice([0, 1], [100, 2]);

// creating one hot encoded y vector.
const IrisY = tf.tensor(
  Array(100)
    .fill([1, 0], 0, 50)
    .fill([0, 1], 50)
);

// normalizing data
const mIrisX = normalizeData(IrisX);

// generate meshgrid
const grid = meshGridRange(
  (range = { x: { min: -2.1, max: +2.1 }, y: { min: -2.1, max: +2.1 } }),
  (division = 50)
);

const tts = trainTestSplit(mIrisX, IrisY, .9);

const trainX = tts[0].x;
const trainY = tts[0].y;

const testX = tts[1].x;
const testY = tts[1].y;

const model = new KDE();

// calculate pNorm for each value of meshgrid

const pNormGrid = grid.map(a => {
  const f = tf
    .tensor(a)
    .transpose()
    .arraySync();
  const w = f.map(b => {
    // return 1*(pNorm(tf.tensor(b).expandDims(1), p=2).flatten().arraySync()[0] )
    return (
      1 *
      model
        .test(
          tf
            .tensor(b)
            .expandDims(1)
            .transpose(),
          trainX,
          (params = { h: 0.05 })
        )
        .flatten()
        .arraySync()[0]
    );
    // return 1*(inducedMatrixNorm(tf.tensor([[1, 2],[0, 2]]),tf.tensor(b).expandDims(1),p=2).flatten().arraySync()[0] )
  });
  return w;
});

// // visualzing the pNorm Grid
const pNormVizData = [
  {
    x: grid[0][0],
    y: grid[0][0],
    z: pNormGrid,
    // type: 'contour',
    type: "surface",

    colorscale: [
      [0, darkModeCols.blue()],
      [0.25, darkModeCols.purple()],
      [0.5, darkModeCols.magenta()],
      [0.75, darkModeCols.yellow()],
      [1, darkModeCols.red()]
    ],

    // contours: {
    //     start: -1,
    //     end: 1,
    //     size: 3
    //  },
    line: {}
  }
];

const layoutSetting = {
  title: "Kernel Density Estimation",
  font: {
    size: 15,
    color: "white",
    family: "Helvetica"
  },
  paper_bgcolor: "#222633"
};

Plotly.newPlot("decisionBoundaryViz", pNormVizData, layoutSetting);
