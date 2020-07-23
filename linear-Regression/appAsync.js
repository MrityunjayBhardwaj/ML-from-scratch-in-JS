import LinearRegression from './Source/Linear-RegressionAsync.mjs';
import * as tfvis from '../dependency/tensorflowJS/tfjs-vis.umd.min.js';

const modelParams = {
  useBias: true,
  nEpoch: 10,
  threshold: 0.01,
  learningRate: 0.01,
};

/* Visualizing */

// input viz
const inputVizObj = new InputViz(
  '#inputSpaceViz',
  {
    width: 750,
    height: 350,
    gridIntervel: { x: 16, y: 8 },
    rangeX: { min: -5, max: 5 },
    rangeY: { min: -2, max: 2 },
    margin: { top: 5, bottom: 10, left: 25, right: 10 },
  },
  true /* spawn points on click */
);
const svg = inputVizObj.getComponents().svg;
svg.style('color', 'white');

// creating the background rect

inputVizObj.getComponents().frame.style('fill', 'white');

// fetching the data points group
const dataPointsGrp = inputVizObj.getComponents().dataPointsGrp;

// creating group for regression line
const afterGridSpace = inputVizObj.getComponents().spaces.afterGridSpace;
const regLineGrp = afterGridSpace.append('g').attr('class', 'regressionLine');

// fetching important info from inputVizObj
const xScale = inputVizObj.getComponents().conversionFns.x;
const yScale = inputVizObj.getComponents().conversionFns.y;
const xInvScale = inputVizObj.getComponents().conversionFns.xInv;
const yInvScale = inputVizObj.getComponents().conversionFns.yInv;

// init the model
let model = null;

// initializing the data Arrays
let trainX = null;
let trainY = null;
/**
 * Updating the visualization
 */
function updateViz() {
  let dataX = [];
  let dataY = [];

  const dataPts = dataPointsGrp._groups[0][0].childNodes;

  for (let i = 0; i < dataPts.length; i++) {
    dataX.push(xInvScale(dataPts[i].attributes.cx.value));
    dataY.push(yInvScale(dataPts[i].attributes.cy.value));
  }

  // sorting dataY realative to the sorted DataX
  [dataX, dataY] = sortAB(dataX, dataY);

  const nSamples = dataPts.length;
  trainX = tf.tensor(dataX).expandDims().transpose();
  trainY = tf.tensor(dataY).expandDims().transpose();
  model = new LinearRegression(
    {
      useBias: useBiasElem.value * 1,
      nEpoch: epochElem.value * 1,
      threshold: thValElem.value * 1,
      learningRate: lrValElem.value * 1,
    },
    callback
  );

  lossArray = [];
  metricArray = [];

  model.fit(trainX, trainY);
}

let lossArray = [];
let metricArray = [];

/**
 * A callback function which gets invoked at the end of each epoch and update our visualization.
 * @param {number} epoch current Epoch of our training loop
 * @param {number} cLoss current loss
 * @param {tf.tensor} cWeights current weight tensor
 * @param {tf.tensor} yPred current predicted output
 * @return {Promise} returns a promise which resolves only when all the visualizations are being updated.
 */
function callback(epoch, cLoss, cWeights, yPred) {
  return new Promise((resolve, reject) => {
    // calculating our predicted y
    const domainPoints = tf.linspace(-5, 5, 100).expandDims().transpose();
    const domainPredY = model.test(domainPoints);

    /* updating our input space visualzer */

    // preparing our regression line data
    const regLineData = domainPredY
      .flatten()
      .arraySync()
      .map((y, i) => {
        return { x: domainPoints.flatten().arraySync()[i], y: y };
      });

    // updating our visualizer
    const regLineSelect = regLineGrp.selectAll('path');
    regLineSelect
      .data([regLineData])
      .enter()
      .append('path')
      .merge(regLineSelect)
      .transition(d3.easeCubic)
      .duration(1)
      .attr(
        'd',
        d3
          .line()
          .x(function (d) {
            return xScale(d.x);
          })
          .y(function (d) {
            return yScale(d.y);
          })
          .curve(d3.curveCardinal)
      )
      .attr('fill', 'none')
      .attr('stroke', darkModeCols.red(1.0))
      .attr('stroke-width', 2.5)
      .on('end', resolve);

    /* visualizing the loss */

    // adding the current loss to our loss array
    lossArray.push(cLoss);

    // logging our current loss
    lossElem.innerHTML = `Loss: ${cLoss.toFixed(3)} | Epoch: ${epoch - 1}`;

    // using tfvis to visualizing our loss trajectory
    const data = {
      values: lossArray.map((val, i) => {
        return { x: i, y: val };
      }),
      series: ['loss'],
    };
    const surface = document.querySelector('#lossViz');
    window.tfvis.render.linechart(surface, data, { width: 400, height: 200 });

    // /* visualizing accuracy */

    // // adding the current metric to our metric array
    // metricArray.push(cWeights.flatten().arraySync());

    // // logging our current metric
    // metricElem.innerHTML = `Accuracy: ${(1-cLoss).toFixed(3)} `;

    // // using tfvis to visualizing our loss trajectory
    // const metricData = { values: metricArray.map((val, i)=>{return {x: val[0], y: val[1]}}), series: ['metric'] }
    // const metricSurface = document.querySelector('#metricViz');
    // window.tfvis.render.linechart(metricSurface, metricData, {width: 400, height: 200, seriesColors: ['orange']});
  });
}

/* Specifing the behaviour of UI elements */

const btnElem = document.getElementById('button');
const lossElem = document.getElementById('loss');
const metricElem = document.getElementById('metric');

btnElem.addEventListener('click', () => {
  console.log(modelParams);
  updateViz();
});

const thresholdElem = document.getElementById('threshold');
const thValElem = document.getElementById('thresholdValue');
thValElem.innerHTML = thresholdElem.value;

thresholdElem.oninput = function () {
  thValElem.value = this.value;
  modelParams.threshold = this.value * 1;
};

thValElem.oninput = function () {
  thresholdElem.value = thValElem.value;
  modelParams.threshold = thValElem.value * 1;
};

const learningRateElem = document.getElementById('learningRate');
const lrValElem = document.getElementById('learningRateValue');
lrValElem.innerHTML = learningRateElem.value;

learningRateElem.oninput = function () {
  lrValElem.value = this.value;
  modelParams.learningRate = this.value * 1;
};

lrValElem.oninput = function () {
  learningRateElem.value = lrValElem.value;
  modelParams.threshold = lrValElem.value * 1;
};

const useBiasElem = document.getElementById('useBias');
const epochElem = document.getElementById('epoch');

useBiasElem.oninput = function () {
  this.value = 1 - this.value * 1;
  modelParams.useBias = this.value * 1;
};

epochElem.oninput = function () {
  modelParams.nEpoch = this.value * 1;
};
