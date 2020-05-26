
let rewardDisVizContainerElem = document.getElementById("my_dataviz")

// set the dimensions and margins of the graph
var margin = {top: 40, right: 30, bottom: 50, left:20},
    width = 460 - margin.left - margin.right,
    height = 400 - margin.top - margin.bottom;

// append the svg object to the body of the page
var svg = d3.select(rewardDisVizContainerElem)
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");


const nArms = 10;

const myData = []

const distMean = [];
for(let i=0;i< nArms;i++){

    distMean.push(  tf.randomNormal([1]).dataSync()[0]);

}

const nSamples = 10;
for(let i=0;i< nSamples;i++){
    const cDataPoint = {};
    for(let i=0;i<nArms;i++){

            cDataPoint[i] =  (distMean[i]+ Math.round(tf.randomNormal([1]).dataSync()[0]) ).toFixed(5);
    }

    myData.push(cDataPoint);
}
// const data = myData;

const savedData = "[{\"0\":\"4.09939\",\"1\":\"0.71420\",\"2\":\"1.26674\",\"3\":\"0.58985\",\"4\":\"-0.03115\",\"5\":\"-0.34572\",\"6\":\"-0.60013\",\"7\":\"-1.75671\",\"8\":\"1.77477\",\"9\":\"-1.45687\"},{\"0\":\"2.09939\",\"1\":\"1.71420\",\"2\":\"-0.73326\",\"3\":\"0.58985\",\"4\":\"-1.03115\",\"5\":\"-0.34572\",\"6\":\"1.39987\",\"7\":\"-1.75671\",\"8\":\"2.77477\",\"9\":\"-2.45687\"},{\"0\":\"2.09939\",\"1\":\"2.71420\",\"2\":\"0.26674\",\"3\":\"1.58985\",\"4\":\"-1.03115\",\"5\":\"-1.34572\",\"6\":\"3.39987\",\"7\":\"-2.75671\",\"8\":\"4.77477\",\"9\":\"-1.45687\"},{\"0\":\"3.09939\",\"1\":\"0.71420\",\"2\":\"-1.73326\",\"3\":\"4.58985\",\"4\":\"-2.03115\",\"5\":\"-0.34572\",\"6\":\"0.39987\",\"7\":\"-0.75671\",\"8\":\"2.77477\",\"9\":\"-0.45687\"},{\"0\":\"3.09939\",\"1\":\"0.71420\",\"2\":\"-0.73326\",\"3\":\"1.58985\",\"4\":\"-2.03115\",\"5\":\"-3.34572\",\"6\":\"1.39987\",\"7\":\"0.24329\",\"8\":\"1.77477\",\"9\":\"-1.45687\"},{\"0\":\"3.09939\",\"1\":\"1.71420\",\"2\":\"-0.73326\",\"3\":\"1.58985\",\"4\":\"-2.03115\",\"5\":\"-2.34572\",\"6\":\"2.39987\",\"7\":\"-0.75671\",\"8\":\"1.77477\",\"9\":\"-2.45687\"},{\"0\":\"4.09939\",\"1\":\"-0.28580\",\"2\":\"-0.73326\",\"3\":\"3.58985\",\"4\":\"-0.03115\",\"5\":\"-2.34572\",\"6\":\"1.39987\",\"7\":\"-3.75671\",\"8\":\"1.77477\",\"9\":\"-1.45687\"},{\"0\":\"2.09939\",\"1\":\"-0.28580\",\"2\":\"1.26674\",\"3\":\"1.58985\",\"4\":\"-1.03115\",\"5\":\"-1.34572\",\"6\":\"-0.60013\",\"7\":\"-1.75671\",\"8\":\"1.77477\",\"9\":\"-0.45687\"},{\"0\":\"1.09939\",\"1\":\"1.71420\",\"2\":\"-1.73326\",\"3\":\"3.58985\",\"4\":\"-0.03115\",\"5\":\"-1.34572\",\"6\":\"-1.60013\",\"7\":\"-0.75671\",\"8\":\"0.77477\",\"9\":\"-2.45687\"},{\"0\":\"2.09939\",\"1\":\"0.71420\",\"2\":\"-1.73326\",\"3\":\"1.58985\",\"4\":\"-2.03115\",\"5\":\"-3.34572\",\"6\":\"0.39987\",\"7\":\"0.24329\",\"8\":\"0.77477\",\"9\":\"-2.45687\"}]";

const data = JSON.parse(savedData);
//read data
// d3.csv("https://raw.githubusercontent.com/zonination/perceptions/master/probly.csv", function(data) {

  // Get the different categories and count them

  var categories = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
  var n = categories.length;

  // Compute the mean of each group
  allMeans = []
  for (i in categories){
    currentGroup = categories[i]
    mean = d3.mean(data, function(d) { return +d[currentGroup] })
    allMeans.push(mean)
  }

  let bgRectGroup = svg.append('g').attr('class', 'bgRect');

  let distGroup = svg.append('g').attr('class', 'rewardDist');
  let distMeanGroup = svg.append('g').attr('class', 'distMean');
  let estActValGroup = svg.append('g').attr("class", "estimatedActionValue");

  // Create a color scale using these means.
  var myColor = d3.scaleSequential()
    .domain([-2,2])
    .interpolator(d3.interpolateRainbow);


    // function myColor(val){

    //   const myCols = [darkModeCols.white(1.0), 
    //                   darkModeCols.darkBlue(1.0), 
    //                   darkModeCols.blue(1.0),
    //                   darkModeCols.green(1.0), 
    //                   darkModeCols.magenta(1.0),
    //                   darkModeCols.purple(1.0),
    //                   darkModeCols.orange(1.0),
    //                   darkModeCols.red(1.0),
    //                   darkModeCols.yellow(1.0),
    //                   darkModeCols.grey(1.0),
    //                 ]

    //                 return (myCols[Math.abs(Math.floor(val))])
    //   // return (myCols[Math.floor(val+3)]);
    // }

  // Add X axis
  var x = d3.scaleLinear()
    .domain([-11, 11])
    .range([ 0, width ]);
  distGroup.append("g")
    .attr("class", "xAxis")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x).tickValues([-4,0, 4]).tickSize(-height) )
    .select(".domain").remove()

  // Add X axis label:
  svg.append("text")
      .attr("text-anchor", "end")
      .attr("x", width/1.3)
      .attr("y", height + 40)
      .text("Reward Distribution");

    svg.append("text")
    .attr("class", "y label")
    .attr("text-anchor", "end")
    .attr("y", -25)
    .attr("x", -100)
    .attr("transform", "rotate(-90)")
    .text("Actions");


  // Create a Y scale for densities
  var y = d3.scaleLinear()
    .domain([0, 1.2])
    .range([ height, 0]);

  // Create the Y axis for names
  var yName = d3.scaleBand()
    .domain(categories)
    .range([0, height])
    .paddingInner(1)
  svg.append("g")
    .call(d3.axisLeft(yName).tickSize(0))
    .select(".domain").remove()

  // Compute kernel density estimation for each column:
  var kde = kernelDensityEstimator(kernelEpanechnikov(7), x.ticks(40)) // increase this 40 for more accurate density.
  var allDensity = []
  for (i = 0; i < n; i++) {
      key = categories[i]
      density = kde( data.map(function(d){  return d[key]; }) )
      allDensity.push({key: key, density: density})
  }

  // Add areas
  distGroup.selectAll("areas")
    .data(allDensity)
    .enter()
    .append("path")
      .attr("transform", function(d){return("translate(0," + (yName(d.key)-height) +")" )})
      .attr("fill", function(d){
        grp = d.key ;
        index = categories.indexOf(grp)
        value = allMeans[index]
        return myColor( value  )
      })
      .datum(function(d){return(d.density)})
      // .attr("opacity", 0.7)
      .attr("stroke", "#000")
      .attr("stroke-width", 0.1)
      .attr("d",  d3.line()
          .curve(d3.curveCardinal)
          .x(function(d) { return x(d[0]); })
          .y(function(d) { return y(d[1]); })
      )

      bgRectGroup.selectAll("rect").data(allDensity).enter()
      .append("rect")
      // .attr("transform", function(d){return("translate(0," + (yName(d.key)-height) +")" )})
      .attr("fill", "none")
      .attr("stroke", "none")
      .attr('stroke-width', '2px')
      .attr("width", width)
      .attr("height", "30px")
      .attr("y",  function(d){return( (yName(d.key))-30  )})
      .attr("rx", "3px")


      distMeanGroup.selectAll("rect").data(allDensity).enter()
      .append("rect")
      // .attr("transform", function(d){return("translate(0," + (yName(d.key)-height) +")" )})
      .attr("fill", "black")
      .attr("stroke", "none")
      .attr('stroke-width', '2px')
      .attr("width", "5px")
      .attr("height", "15px")
      .attr("y",  function(d){return( (yName(d.key))-15 )})
      .attr("x", (_,i) =>{ return x(allMeans[i]) })
      .attr("rx", "2px")

      estActValGroup.selectAll("rect").data(allDensity).enter()
      .append("rect")
      // .attr("transform", function(d){return("translate(0," + (yName(d.key)-height) +")" )})
      .attr("fill", "black")
      .attr('stroke-width', '1px')
      .attr("width", "2px")
      .attr("height", "10px")
      .attr("y",  function(d){return( (yName(d.key))-10 )})
      .attr("x", (_,i) =>{ return x(0) })
      .attr("rx", "2px")
      // .attr("opacity", .7);
// svg.selectAll('')


// This is what I need to compute kernel density estimation
function kernelDensityEstimator(kernel, X) {
  return function(V) {
    return X.map(function(x) {
      return [x, d3.mean(V, function(v) { return kernel(x - v); })];
    });
  };
}
function kernelEpanechnikov(k) {
  return function(v) {
    return Math.abs(v /= k) <= 1 ? 0.75 * (1 - v * v) / k : 0;
  };
}


function updateRewardDistViz(action, estActionVal, highlightInterval=1000){

 
  rewardDisVizContainerElem.style.borderColor="red"
  setTimeout(()=>{

  rewardDisVizContainerElem.style.borderColor=""
  }, highlightInterval)

  let bgRectSelect = bgRectGroup.selectAll('rect');

  bgRectSelect.datum([1]).enter()
  .append('rect').merge(bgRectSelect)
  .attr('stroke', (d,i) =>{ return (i === action)?  'rgb(240, 14,14)' : 'none'})

  let estActValSelect = estActValGroup.selectAll('rect');

  // estActValSelect.data(estActionVal).enter()
  // .append('rect').merge(estActValSelect)
  // .attr('x', (d,i)=>{
  //   return x(d)
  // })
  // .attr('fill', (_,i)=>{ if(i === action)return'red'; return 'blue' } )
  // .attr('width', (_,i)=>{ if(i === action)return'red' } )

}

updateRewardDistViz(3)
function resetRewardDistViz(){

  let bgRectSelect = bgRectGroup.selectAll('rect');

  bgRectSelect.datum([1]).enter()
  .append('rect').merge(bgRectSelect)
  .attr('stroke', (d,i) =>{ return 'none'})
}