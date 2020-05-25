// set the dimensions and margins of the graph
var margin = {top: 10, right: 30, bottom: 30, left: 20},
    timelineVizWidth = 470 - margin.left - margin.right,
    timelineVizHeight = 180 - margin.top - margin.bottom;

let timelineVizElem = document.getElementById("timelineViz")
// append the svg object to the body of the page
var timelineSvg = d3.select(timelineVizElem)
  .append("svg")
    .attr("width", timelineVizWidth + margin.left + margin.right)
    .attr("height", timelineVizHeight + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");


let rewardGroup = timelineSvg.append('g').attr('class', 'rewardTimeline');
let xAxisGroup = timelineSvg.append('g').attr('class', 'axisX');

const nSteps = 10;
let rewardArray = [];

// rewardArray = tf.randomNormal([100]).flatten().arraySync();

let rewardData = rewardArray.map((d,i)=>{ return {x: i,y:d}})

//Read the data
// d3.csv("https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/data_IC.csv",function(data) {

  // Add X axis --> it is a date format
  var timelineX = d3.scaleLinear()
    .domain([1,nSteps])
    .range([ 0, timelineVizWidth ]);
  xAxisGroup.append("g")
    .attr("transform", "translate(0," + timelineVizHeight + ")")
    .call(d3.axisBottom(timelineX));

  // Add Y axis
  var timelineY = d3.scaleLinear()
    .domain([-8, 8])
    .range([ timelineVizHeight, 0 ]);
  timelineSvg.append("g")
    .call(d3.axisLeft(timelineY).ticks(5))

  // This allows to find the closest X index of the mouse:
  var bisect = d3.bisector(function(d) { return d.x; }).left;

  // Create the circle that travels along the curve of chart
  var focus = timelineSvg
    .append('g')
    .append('circle')
      .style("fill", "none")
      .attr("stroke", "black")
      .attr('r', 8.5)
      .style("opacity", 0)

  // Create the text that travels along the curve of chart
  var focusText = timelineSvg
    .append('g')
    .append('text')
      .style("opacity", 0)
      .attr("text-anchor", "left")
      .attr("alignment-baseline", "middle")

       // Add the line
  rewardGroup 
  .append("path")
  .datum(rewardData)
  .attr("fill", "none")
  .attr("stroke", "steelblue")
  .attr("stroke-timelineVizWidth", 1.5)
  .attr("d", d3.line()
    .x(function(d) { return timelineX(d.x) })
    .y(function(d) { return timelineY(d.y) })
    )

  // Create a rect on top of the timelineSvg area: this rectangle recovers mouse position
  timelineSvg
    .append('rect')
    .style("fill", "none")
    .style("pointer-events", "all")
    .attr('timelineVizWidth', timelineVizWidth)
    .attr('timelineVizHeight', timelineVizHeight)
    .on('mouseover', mouseover)
    .on('mousemove', mousemove)
    .on('mouseout', mouseout);


  // What happens when the mouse move -> show the annotations at the right positions.
  function mouseover() {
    focus.style("opacity", 1)
    focusText.style("opacity",1)
  }

  function mousemove() {
    // recover coordinate we need
    var x0 = x.invert(d3.mouse(this)[0]);
    var i = bisect(rewardData, x0, 1);
    selectedData = rewardData[i]
    focus
      .attr("cx", timelineX(selectedData.x))
      .attr("cy", timelineY(selectedData.y))
    focusText
      .html("x:" + selectedData.x + "  -  " + "y:" + selectedData.y)
      .attr("x", timelineX(selectedData.x)+15)
      .attr("y", timelineY(selectedData.y))
    }
  function mouseout() {
    focus.style("opacity", 0)
    focusText.style("opacity", 0)
  }

// })

function updateTimeline(rewardArray, highlightInterval=1000){
   
    timelineVizElem.style.borderColor="red";
    setTimeout(()=>{


        timelineVizElem.style.borderColor="";
    }, highlightInterval);

    const pathSelect = rewardGroup.selectAll('path');

    const r = rewardArray.map((d,i)=>{ return {x: i+1,y:d}})

  timelineX = d3.scaleLinear()
    .domain([1,Math.max(nSteps,rewardArray.length)])
    .range([ 0, timelineVizWidth ]);

    let selectG = xAxisGroup.selectAll('g')
    
    selectG.data([""]).enter().append('g')
    .merge(selectG)
    .call(d3.axisBottom(timelineX));

//   xAxisGroup.append('g')
//     .attr("transform", "translate(0," + 110 + ")")
//     .call(d3.axisBottom(timelineX));

    // console.log(rewardArray)
    pathSelect 
    .datum(r).enter()
    .append("path")
    .merge(pathSelect)
    .attr('stroke', darkModeCols.green(1.0))
    .attr("stroke-width", 1.2)
    .attr("d", d3.line()
        .x(function(d, i) {
            return timelineX(d.x) })
        .y(function(d) { return timelineY(d.y) })
        )


}