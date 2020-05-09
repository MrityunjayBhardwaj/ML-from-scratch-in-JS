
let mySVM = 1;

let isFirst = 0;
let currClass = 0;

let useSaved = 0;

let canUpdate = 1;

const polyDegreeElem = document.getElementById('polyDegree');
const rbfSigmaElem = document.getElementById('rbfSigma');
const tolleranceElem = document.getElementById('tollerance');

let kernelType = 'rbf';

function click(){
  // Ignore the click event if it was suppressed
  // if (d3.event.defaultPrevented) return;

  // Extract the click location\    
  var point = d3.mouse(this)
  , p = {x: point[0], y: point[1] };

    console.log('yes')

    const margin = inputVizObj.getComponents().svgSettings.margin;
    const svg = inputVizObj.getComponents().svg;

  // Append a new point to our data Points group
  svg.select((currClass)? '.dataPoints1' : '.dataPoints0')
        .append('circle')
        .attr('cx', (p.x -margin.left))
        .attr('cy', (p.y -margin.top))
          .attr('r', '7')
          .style('fill',(currClass)? "rgb(200,0,0)" : "blue")
          .call(drag)

  if(isFirst){
    // updateViz();
  }
  else{

    // if (currClass)
    isFirst =1;
  }
}


// Define drag beavior
var drag = d3.drag()
    .on("drag", dragmove);

function dragmove(d) {
  var x = d3.event.x;
  var y = d3.event.y;
  d3.select(this)
      .attr('cx', (x))
      .attr('cy', (y))

      if (canUpdate){

        let updatePromise = updateViz();

        console.log(updatePromise);
        canUpdate = 0;
        updatePromise.then((a) =>{
          console.log('from UpdatePromise', a);
        })
      }
      else{
        console.log('still Updating!!!!')
      }
}

// TODO: Fix the on Click event mechanism here as well as in the inputViz function in utils.js
// creating inputViz 
const inputVizObj = new inputViz('#inputSpace',{isGrid: false, isAxisLine: false}, eventHandlers={onClick: click});

const lossSpaceRange = {min: -3, max: 3};

// creating loss Landscape Viz 
const lossVizObj = new inputViz('#lossLandscape',{isGrid: true, isAxisLine: true, rangeX: lossSpaceRange,rangeY: lossSpaceRange} );


// const svg = inputVizObj.getComponents().svg;
const svg = inputVizObj.getComponents().spaces.afterGridSpace;

const chartDecorGroup = svg.append('g').attr('class', 'chartDecor');
const chartDataGroup  = svg.append('g').attr('class', 'chartData');



/* adding a data Point group of class 0 in our svg container */
inputVizObj.getComponents().svg.append('g')
  .attr('class', 'dataPoints0')

/* adding a data Point group of class 1 in our svg container */
inputVizObj.getComponents().svg.append('g')
  .attr('class', 'dataPoints1')

/* adding a group to show weather a point is a support vector or not */
const chartSVGroup = svg.append('g')
  .attr('class', 'supportVectors')



  /* Loss Landscape groups */
const lossSvg = lossVizObj.getComponents().spaces.afterGridSpace;





const xRange0 = {min: -10, max: 10 };
const xRange1 = {min: -10, max: 10 };


const svgElemSize = 500;

const margin = inputVizObj.getComponents().svgSettings.margin;

// creating chart position properties:-

const innerHeight = inputVizObj.getComponents().svgSettings.width;
const innerWidth = inputVizObj.getComponents().svgSettings.height;

let dataPoints = chartDataGroup.append('g').attr('class', 'dataPoints');
let heatmap = chartDataGroup.append('g').attr('class', 'heatMap');
let contourMap = chartDataGroup.append('g').attr('class', 'contourMap');
let linearDB = chartDataGroup.append('g').attr('class', 'linearDB');

let lossContourMap = lossSvg.append('g').attr('class', 'lossContourMap');

// group selection
let svSelect = chartSVGroup
.selectAll('circle');


const xScale = inputVizObj.getComponents().conversionFns.x;
const yScale = inputVizObj.getComponents().conversionFns.y;

const xScaleL = lossVizObj.getComponents().conversionFns.x;
const yScaleL = lossVizObj.getComponents().conversionFns.y;

heatmap
.append('rect')
.attr('fill', 
    darkModeCols.blue(1.0)
    )
.attr('width', innerWidth)
.attr('height', innerHeight)

const axisGroup = chartDecorGroup.append('g').attr('class', 'axis');


// lossSvg
// .append('rect')
// .attr('fill', 
//     darkModeCols.blue(1.0)
//     )
// .attr('width', innerWidth)
// .attr('height', innerHeight)

/* adding a data Point group of class 0 in our svg container */
let lossWeights = lossSvg.append('g')
  .attr('class', 'weights')


let isP = 0;


function givePromise(){

  return new Promise((resolve, reject) =>{
    setTimeout(function(){
      resolve('awesome!')
    }, 1000)
  }).then(
    (g) =>{

      console.log(g);

      isP = 1;

    }
  )
} 
let p  = givePromise();


  let perEpochWeightVector = [];

function updateVisuals(model){


    let calcWeights = 0;
    let calcBias = 0;
    let calcAlphas = 0;
    let calcDataX = 0; // all the data points that are support vectors

    perEpochWeightVector = [];

    if (mySVM){
      calcWeights = model.getParams().weights;
      calcBias    = model.getParams().bias;
      calcAlphas = model.getAlphas().flatten().arraySync();
      calcDataX = model.getData().x.arraySync();

    }else{

      calcWeights = tf.tensor( model.getWeights().w ).expandDims();
      calcBias = tf.tensor( model.getWeights().b ).reshape([1,1]);
      calcAlphas = tf.tensor( model.alpha ).expandDims().transpose();
    }

  perEpochWeightVector.push(calcWeights.flatten().arraySync());





let hyperplaneViz = calcHyperplane(model, {
  weights: calcWeights,
  bias: calcBias,
  range: [
  
    xRange0,
    xRange1
  ],

})

// heatmap
// let rectSelect = heatmap.selectAll('rect')

// rectSelect
// .data(hyperplaneViz.heatMap.x).enter()
// .append('rect')
// .merge(rectSelect)
// .attr('transform', (d) => `translate(${xScale(d[0])},${yScale(d[1])}) `)
// .attr('fill', (d,i) => {
//     if (hyperplaneViz.heatMap.y[i] > 0)return darkModeCols.red(1.0);
//     if (hyperplaneViz.heatMap.y[i] < 0)return darkModeCols.blue(1.0);
//     return 'white';
// })
// .attr('stroke', (d,i) => (hyperplaneViz.heatMap[i] === 0)? 'gray' : 'none')
// .attr('width', 19)
// .attr('height', 19)


// visualizing support Vectors
svSelect = chartSVGroup
.selectAll('circle');

svSelect
.data(calcDataX).enter()
.append('circle')
.merge(svSelect)
        .attr('cx',(d, _) => xScale(d[0]) )
        .attr('cy',(d, _) => yScale(d[1]) )
        .attr('stroke', (_,i) => {return (calcAlphas[i] > 0)? 'black' : 'none' } )
  .attr('stroke-dasharray', (d,i) =>  "10 5")
  .attr("stroke-linejoin", "round")
        .attr('fill', 'none')
        .attr('stroke-width', 2)
          .attr('r', '12')



  /* plotting contour Map */
  let pathSelect = contourMap.selectAll('path');

  if(kernelType == 'linear'){

    // remove the contour plot of other kernels
    pathSelect
    .remove();

  }else{

    // compute the density data
    var densityData = d3.contourDensity()
    .x(function(d) { return xScale(d[0]); })   // x and y = column name in .csv input data
    .y(function(d) { return yScale(d[1]); })
    .size([innerWidth, innerHeight])
    .thresholds(5)
    .bandwidth(20)    // smaller = more precision in lines = more lines
    (tf.tensor(hyperplaneViz.heatMap.x).concat(tf.tensor(hyperplaneViz.heatMap.y).expandDims().transpose(), 1).arraySync() .filter((d)=> { return (d[2] > 0)} ) )
    // Add the contour: several "path"


    pathSelect
    .data(densityData).enter()
    .append("path")
    .merge(pathSelect)
      .attr("d", d3.geoPath())
      .attr("fill", (d,i) => { if(i === 1)return darkModeCols.red(1.0); return "none"})
      .attr("stroke-width",(kernelType === "linear")? "0" :  "2")
      .attr("stroke", (d,i)=>{ 
        if (kernelType === 'linear') return "none";

        if(i === 0) return "black"
        if(i === 1) return "black"
        if(i === 2) return "black"

      } )
      .attr('stroke-dasharray', (d,i) => {if(i != 1)return "10 7"})
      .attr("stroke-linejoin", "round")

  }
// adding hyperplane path
let hyperplanePathSelect = linearDB.selectAll('path');

if(kernelType === 'linear'){

  // plotting a hyperplane
  const lineGen = d3.line();
  const hypNormalLinePath = lineGen(hyperplaneViz.hypNormal.map(d =>  [xScale(d.x), yScale(d.y)]) ); 
  const hyperplaneLinePath = lineGen(hyperplaneViz.hyperplane.map(d =>  [xScale(d.x), yScale(d.y)]) );
  const marginLeftLinePath = lineGen(hyperplaneViz.marginLeft.map(d => [xScale(d.x), yScale(d.y)] ));
  const marginRightLinePath = lineGen(hyperplaneViz.marginRight.map(d => [xScale(d.x), yScale(d.y)] ));

  const linearContourPlot = lineGen(hyperplaneViz.contourPlot.map(d => [xScale(d.x), yScale(d.y)] ))
  // const linearContourPlot = lineGen(hyperplaneViz.hyperplane.concat([{x: 10, y: -10}, {x: 10, y: 10}]).map(d => [xScale(d.x), yScale(d.y)] ));



  // visualizing left and right margin
  hyperplanePathSelect
  .data([linearContourPlot, hyperplaneLinePath, marginLeftLinePath, marginRightLinePath]).enter()
  .append("path")
  .merge(hyperplanePathSelect)
    .attr('d', (d)=> d)
    .attr('stroke', 'black')
    .attr('stroke-width', 2)
    .attr('stroke-dasharray', (d,i) => {if(i != 1 && i != 0)return "10 7"})
    .attr("stroke-linejoin", "round")
    .attr('fill', (_,i)=>{ return (i === 0)? darkModeCols.red(1.0) : 'none'});




  let weightsSelect = lossWeights.selectAll('circle');

  const weightLine = lineGen(perEpochWeightVector.map(d => [xScaleL(d[0]), yScaleL(d[1])] ));

  console.log(perEpochWeightVector)
  weightsSelect
  .data(perEpochWeightVector).enter()
  .append('circle')
  .merge(weightsSelect)
          .attr('cx',(d, _) => xScaleL(d[0]) )
          .attr('cy',(d, _) => yScaleL(d[1]) )
          .attr('stroke', darkModeCols.purple(1.0))
            .attr('r', '5') 

  let weightsSelectPath = lossWeights.selectAll('path');
    weightsSelectPath
    .data([weightLine]).enter()
    .append("path")
    .merge(weightsSelectPath)
      .attr('d', (d)=> d)
      .attr('stroke', darkModeCols.red(1.0))
      .attr('stroke-width', 3)
      .attr("stroke-linejoin", "round")
      .attr('fill', 'none');


  // calcDataX.


  }else{
    hyperplanePathSelect.remove();
  }




};


// after finishing the calculation allow updating...
function onFinish(){

  canUpdate = 1;
}



























let model = new svm(); 

/**
 *  here, we will update our classification visualizer
 */
async function updateViz(useSaved=false){

    if (!mySVM){
      model = new svmjs.SVM(); 
    }


  const dataPoints0 = [];
  const dataPoints1 = [];

  // extract the data points svg object
  const pointsGroup0 = d3.selectAll('.dataPoints0')._groups[0][0].childNodes;
  const pointsGroup1 = d3.selectAll('.dataPoints1')._groups[0][0].childNodes;

  for(let i=0; i<pointsGroup0.length;i++){
    const currPoint = [d3.select(pointsGroup0[i]).attr('cx')*1, d3.select(pointsGroup0[i]).attr('cy')*1]

    // converting these corrdinates back to the data range2
    currPoint[0] = inputVizObj.getComponents().conversionFns.xInv(currPoint[0]);
    currPoint[1] = inputVizObj.getComponents().conversionFns.yInv(currPoint[1]);

    dataPoints0.push(currPoint)
  }

  for(let i=0; i<pointsGroup1.length;i++){
    const currPoint = [d3.select(pointsGroup1[i]).attr('cx')*1, d3.select(pointsGroup1[i]).attr('cy')*1]

    // converting these corrdinates back to the data range
    currPoint[0] = inputVizObj.getComponents().conversionFns.xInv(currPoint[0]);
    currPoint[1] = inputVizObj.getComponents().conversionFns.yInv(currPoint[1]);

    dataPoints1.push(currPoint)
  }
  const dataPoints0_y = new Array(pointsGroup0.length).fill([-1]);
  const dataPoints1_y = new Array(pointsGroup1.length).fill([+1]);

  let data = dataPoints0.concat(dataPoints1);

  let trainData = {x: tf.tensor(data), 
                   y: tf.tensor(dataPoints0_y).concat(tf.tensor(dataPoints1_y), 0)}




    const kernelParams = {'linear': {}, 
                          'poly': {degree: polyDegreeElem.value*1}, 
                          'rbf': {sigma: rbfSigmaElem.value*1}};


    canUpdate = 0;


    /* CALCULATING THE LOSS LANDSCAPE */

    const division = 20;
    const dims = 3;
    const inpMeshGridTensor = tf.tensor(meshTensor(lossSpaceRange.min, lossSpaceRange.max, division, dims)).reshape([division**dims, dims]);


    // for(let i=0;i< inp0.length*inp1.length;i++){

       let {objFn: objFnGrid,constraints: constraintRegion} = ( model.calcObjectiveFn(
                                              inpMeshGridTensor,
                                              trainData.x,
                                              trainData.y, 
                                              kernelType, 
                                              kernelParams[kernelType]) );

    // }


    // objFnGrid = objFnGrid.reshape([inp0.length, inp1.length]);

    /* visualizing the loss landscape */

    objFnGrid = objFnGrid.concat(constraintRegion, axis=1);

let rectSelect = lossContourMap.selectAll('rect')

let intensityRange = {min: tf.min(objFnGrid).dataSync()[0], max: tf.max(objFnGrid).dataSync()[0], }
const colorZ = [

darkModeCols.red(1),
darkModeCols.green(1),
darkModeCols.blue(1),
darkModeCols.yellow(1),
darkModeCols.magenta(1),
]

const th = [.001,0.01,0.1, 0.5,1]
function condition(v){

  if(v < th[0])return colorZ[0];
  if(v < th[1])return colorZ[1];
  if(v < th[2])return colorZ[2];
  if(v < th[3])return colorZ[3];

  return colorZ[4];
}

  // dim reduction of inpMeshGrid
  const pca = new myPCA();
  pca.fit(inpMeshGridTensor);

  const reducedInpMeshGrid = pca.dimReduction(2);

    objFnGrid = reducedInpMeshGrid.concat(objFnGrid,axis=1)


rectSelect
.data(objFnGrid.arraySync()).enter()
.append('rect')
.merge(rectSelect)
.attr('transform', (d) => `translate(${xScaleL(d[0])},${yScaleL(d[1])}) `)
.attr('fill', (d,_) => {
  // return condition(d[2])
  const currIntensity = d[2];
  const normIntensity =  ( (currIntensity - intensityRange.min)/intensityRange.max );

  return (d[3] || 1)?  `rgb(${normIntensity*255}, ${normIntensity*255}, ${normIntensity*255} )` : darkModeCols.green(1.0);
})
// .attr('stroke', (_,i) => (hyperplaneViz.heatMap[i] === 0)? 'gray' : 'none')
.attr('width', 19)
.attr('height', 19)

  //   // combining the coordinates and the objFn Values

  //   // compute the density data
  //   var densityDataL = d3.contourDensity()
  //   .x(function(d) { return xScaleL(d[0]); })   // x and y = column name in .csv input data
  //   .y(function(d) { return yScaleL(d[1]); })
  //   .size([division, division])
  //   .thresholds([0,.001,.01,.1,.2,,3,.4,.5])
  //   // .bandwidth(20)    // smaller = more precision in lines = more lines
  //   // (tf.tensor(hyperplaneViz.heatMap.x).concat(tf.tensor(hyperplaneViz.heatMap.y).expandDims().transpose(), 1).arraySync() .filter((d)=> { return (d[2] > 0)} ) )
  //   (objFnGrid.arraySync())
  //   // Add the contour: several "path"


  // let lossPathSelect = lossContourMap.selectAll('path');

  //   lossPathSelect
  //   .data(densityDataL).enter()
  //   .append("path")
  //   .merge(lossPathSelect)
  //     .attr("d", d3.geoPath())
  //     .attr("fill", (d,i) => {return `rgba(${55+Math.random()*200}, ${55+Math.random()*200}, ${55+Math.random()*200}, 1)`})
  //     .attr("stroke-width",(kernelType === "linear")? "2" :  "2")
  //     .attr("stroke", (d,i)=>{ 
  //       // if (kernelType === 'linear') return "none";

  //       return "black"
  //       // if(i === 0) return "black"
  //       if(i === 1) return "black"
  //       if(i === 2) return "black"

  //     } )
  //     .attr('stroke-dasharray', (d,i) => {return "10 7"})
  //     .attr("stroke-linejoin", "round")






    model.fit(trainData, 
      params = {threshold: .01, 
                tollerance:  tolleranceElem.value*1, 
                epoch: 100,
                useSaved: 0, 
                kernelType: kernelType, 
                kernelParams: kernelParams[kernelType],
                verbose: true, 
                onFinishCallback:onFinish, 
                onEpochCallback: updateVisuals
              })

}


function changeClass(){

    currClass = 1-currClass;
}


const polyOptions = document.getElementById('polyOptions');
const rbfOptions = document.getElementById('rbfOptions');
function showPolyOptions(){
 polyOptions.style.display = ''; 
 rbfOptions.style.display = 'none'; 

 svSelect.remove();

 kernelType = 'poly';
}

function showRbfOptions(){
 rbfOptions.style.display = ''; 
 polyOptions.style.display = 'none'; 

 svSelect.remove();

 kernelType = 'rbf';
}

function showLinearOptions(){
 rbfOptions.style.display  = 'none'; 
 polyOptions.style.display = 'none'; 

 svSelect.remove();

  kernelType = 'linear';
}