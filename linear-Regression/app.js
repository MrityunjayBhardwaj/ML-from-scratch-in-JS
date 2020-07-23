import LinearRegression from './Source/Linear-Regression.mjs';
// import * as tfvis from '../dependency/tensorflowJS/tfjs-vis.umd.min.js';
// import * as tf from '../dependency/tensorflowJS/tf.es2017.js';
// import * as tf from '@tensorflow/tfjs';
// import * as tfvis from '@tensorflow/tfjs-vis';

// const { default: LinearRegression } = require("./Source/Linear-Regression");

// Generate some synthetic data for Regression.
let trainX = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
let trainY = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);
const modelParams = {useBias: true, 
        nEpoch: 10, 
        threshold: 0.01, 
        learningRate: 0.01, };

let matrixX = tf.tensor([[1,2],[2,5],[7,8],[12,15]]);

const weights = tf.tensor([3,2,-3]).expandDims(1);

const matrixXMod = matrixX.concat( tf.ones([matrixX.shape[0],1]), 1);
matrixX.print();

let matrixY = matrixXMod.matMul( weights );

/* Visualizing */
// fetching the DOM elems
const dataSpaceVizElem   = document.querySelector('#inputSpaceViz');
const lossVizElem   = document.querySelector('#lossViz');
const metricVizElem = document.querySelector('#metricViz');

/**
 * function for a click event
 */
function click() {
  // Ignore the click event if it was suppressed
  // if (d3.event.defaultPrevented) return;

  // Extract the click location\
  const point = d3.mouse(this);
  const p = { x: point[0], y: point[1] };

  console.log('yes');

  // Append a new point to our data Points group
  svg
    .select('.dataPoints')
    .append('circle')
    .attr('cx', p.x - margin.left)
    .attr('cy', p.y - margin.top)
    .attr('r', '7')
    .style('fill', darkModeCols.blue(1.0))
    .call(drag);
}

// Define drag beavior
const drag = d3.drag().on('drag', dragmove);

/**
 * drag the data point
 */
function dragmove() {
  const x = d3.event.x;
  const y = d3.event.y;
  d3.select(this).attr('cx', x).attr('cy', y);
}

// input viz
const inputVizObj = new inputViz(
  '#inputSpaceViz',
  {
    width: 750,
    height: 350,
    gridIntervel: { x: 16, y: 8 },
    rangeX: { min: -5, max: 5 },
    rangeY: { min: -2, max: 2 },
  },
   { onClick: click } /* event handler */
);

const svg = inputVizObj.getComponents().svg;
svg.style('color', 'white');

const margin = inputVizObj.getComponents().svgSettings.margin;

// creating the background rect
inputVizObj
  .getComponents()
  .spaces.beforeGridSpace.append('rect')
  .attr('width', inputVizObj.getComponents().svgSettings.width)
  .attr('height', inputVizObj.getComponents().svgSettings.width)
  .style('fill', 'white');

  inputVizObj.getComponents().frame.style('fill', 'white')

const afterGridSpace = inputVizObj.getComponents().spaces.afterGridSpace;

const regLineGrp = afterGridSpace.append('g').attr('class' , 'regressionLine');
// create group for data points
const dataPointsGrp = svg.append('g').attr('class', 'dataPoints');

// fetching important info from inputVizObj
const xScale = inputVizObj.getComponents().conversionFns.x;
const yScale = inputVizObj.getComponents().conversionFns.y;

const xInvScale = inputVizObj.getComponents().conversionFns.xInv;
const yInvScale = inputVizObj.getComponents().conversionFns.yInv;

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

    // sort
    [dataX, dataY] = sortAB(dataX, dataY);

    const nSamples = dataPts.length;
    const trainX = tf.tensor(dataX).expandDims().transpose();
    const trainY = tf.tensor(dataY).expandDims().transpose();
    const model = new LinearRegression(
      modelParams
    );
      

    model.fit(trainX, trainY)
    const predY = model.test(trainX);

    const domainPoints = tf.linspace(-5, 5, 100).expandDims().transpose();
    console.log('testing model')
    const domainPredY = model.test(domainPoints);

    const regLineData = domainPredY.flatten().arraySync().map((y, i) => {return {x: (domainPoints.flatten().arraySync()[i]), y: (y)}});

    console.log(domainPoints.length, domainPredY.print())

    const regLineSelect = regLineGrp.selectAll('path')

    regLineSelect.data([regLineData]).enter().append('path')
    .merge(regLineSelect)
    .transition(d3.easeCubic)
    .duration(500)
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
    .attr('stroke-width', 2.5);

    lossElem.innerHTML = `Loss: ${tf.losses.meanSquaredError(predY, trainY ).flatten().arraySync()[0]}`
}

const btnElem = document.getElementById('button');
const lossElem = document.getElementById('loss');

btnElem.addEventListener('click', ()=>{
  console.log(modelParams);
    updateViz();
})

const thresholdElem = document.getElementById("threshold");
const thValElem = document.getElementById("thresholdValue");
thValElem.innerHTML = thresholdElem.value;

thresholdElem.oninput = function() {
  thValElem.innerHTML = this.value;
  modelParams.threshold = this.value*1;
}

const learningRateElem = document.getElementById("learningRate");
const lrValElem = document.getElementById("learningRateValue");
lrValElem.innerHTML = learningRateElem.value;

learningRateElem.oninput = function() {
  lrValElem.innerHTML = this.value;

  modelParams.learningRate = this.value*1;
}


  const useBiasElem = document.getElementById('useBias');
  const epochElem = document.getElementById('epoch');

  useBiasElem.oninput = function() {
    modelParams.useBias = this.value*1;
  }

  epochElem.oninput = function() {
    modelParams.nEpoch = this.value*1;
  }
