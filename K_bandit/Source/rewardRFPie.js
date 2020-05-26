// set the dimensions and margins of the graph
var width = 350
    height = 300
    margin = 40

// The radius of the pieplot is half the width or half the height (smallest one). I subtract a bit of margin.
var radius = Math.min(width, height) / 2 - margin

const rewardRFPieVizElem = document.getElementById("rewardRFPieViz")
// append the rewardRFPieSvg object to the div called 'my_dataviz'
var rewardRFPieSvg = d3.select(rewardRFPieVizElem)
  .append("svg")
    .attr("width", width)
    .attr("height", height)
  .append("g")
    .attr("transform", "translate(" + width / 2 + "," + height / 2 + ")");

let rewardRFPieGroup = rewardRFPieSvg.append('g').attr('class', 'rewardRFPie');

// Create dummy data
var rewardRFPieData = {0: 0, 1: 0, 2:0, 3:0, 4:2, 5:0, 6: 2,7:2, 8: 9, 9: 9}

// set the color scale
var color = d3.scaleOrdinal()
  .domain(rewardRFPieData)
//   .range(["#98abc5", "#8a89a6", "#7b6888", "#6b486b", "#a05d56"])

// Compute the position of each group on the pie:
var pie = d3.pie()
  .padAngle(.02)
  .value(function(d) {return d.value; }).sort(null);
var data_ready = pie(d3.entries(rewardRFPieData))

// The arc generator
var arc = d3.arc()
    .innerRadius(40)         // This is the size of the donut hole
    .outerRadius(radius)
    .cornerRadius(3);

// Another arc that won't be drawn. Just for labels positioning
var outerArc = d3.arc()
    .innerRadius(60)         // This is the size of the donut hole
    .outerRadius(radius*1.8)

console.log(data_ready, rewardRFPieData)
// Build the pie chart: Basically, each part of the pie is a path that we build using the arc function.
rewardRFPieGroup
  .selectAll('path')
  .data(data_ready)
  .enter()
  .append('path')
  .attr('d', arc
  )
  .attr('fill', function(d){ return(myColor(allMeans[d.data.key])) })
  .attr("stroke", "none")
  .style("stroke-width", "1px")
  // .style("opacity", 0.7)

// Add the polylines between chart and labels:
rewardRFPieGroup
  .selectAll('allPolylines')
  .data(data_ready)
  .enter()
  .append('polyline')
    .attr("stroke", "white")
    .style("fill", "none")
    .attr("stroke-width", 1)
    .attr('points', function(d) {
      var posA = arc.centroid(d) // line insertion in the slice
      var posB = outerArc.centroid(d) // line break: we use the other arc generator that has been built only for that
      var posC = outerArc.centroid(d); // Label position = almost the same as posB
      var midangle = d.startAngle + (d.endAngle - d.startAngle) / 2 // we need the angle to see if the X position will be at the extreme right or extreme left
      posC[0] = radius * 0.95 * (midangle < Math.PI ? 1 : -1); // multiply by 1 or -1 to put it on the right or on the left
      return [posA, posB, posC]
    })

// Add the polylines between chart and labels:
rewardRFPieGroup
  .selectAll('allLabels')
  .data(data_ready)
  .enter()
  .append('text')
    .text( function(d) { console.log(d.data.key) ; return d.data.key } )
    .attr('transform', function(d) {
        var pos = outerArc.centroid(d);
        var midangle = d.startAngle + (d.endAngle - d.startAngle) / 2
        pos[0] = radius * 0.99 * (midangle < Math.PI ? 1 : -1);
        return 'translate(' + pos + ')';
    })
    .style('text-anchor', function(d) {
        var midangle = d.startAngle + (d.endAngle - d.startAngle) / 2
        return (midangle < Math.PI ? 'start' : 'end')
    })
    .attr('font-size',25)


function updateRewardRFPie(cAction,array, highlightInterval=1000){


    rewardRFPieVizElem.style.borderColor="red"; 
    setTimeout(()=>{

        rewardRFPieVizElem.style.borderColor="";
    }, highlightInterval);

    // array = normalize(array);
    rewardRFPieData = array.map((d,i)=>{let a = {}; a[i] = d; return a});

    rewardRFPieData = {}
    for(let i=0;i< array.length;i++){
        rewardRFPieData[i] = array[i];
    }

    let rewardRFPiePathSelect = rewardRFPieGroup.selectAll("path");
    let rewardRFPiePolyLineSelect = rewardRFPieGroup.selectAll("polyline");
    let rewardRFPieTextSelect = rewardRFPieGroup.selectAll("text");

    rewardRFPieData = pie(d3.entries(rewardRFPieData));

    // console.log(rewardRFPieData, rewardRFScaleX(4), rewardRFScaleY(5))
    rewardRFPiePathSelect.data(rewardRFPieData).enter()
    .append('path').merge(rewardRFPiePathSelect)
    .attr('d', arc
    )
    .attr('stroke', function(d, i){if(i === cAction)return 'red'; return 'none' })
    .style("stroke-width", "3px")


// Add the polylines between chart and labels:
rewardRFPiePolyLineSelect
  .data(rewardRFPieData)
  .enter()
  .append('polyline')
  .merge(rewardRFPiePolyLineSelect)
    .attr("stroke", "white")
    .style("fill", "none")
    .attr("stroke-width", 1.5)
    .attr('points', function(d) {
      var posA = arc.centroid(d) // line insertion in the slice
      var posB = outerArc.centroid(d) // line break: we use the other arc generator that has been built only for that
      var posC = outerArc.centroid(d); // Label position = almost the same as posB
      var midangle = d.startAngle + (d.endAngle - d.startAngle) / 2 // we need the angle to see if the X position will be at the extreme right or extreme left
      // posC[0] = radius * 0.95 * (midangle < Math.PI ? 1 : -1); // multiply by 1 or -1 to put it on the right or on the left
      return [posA, posB, posC]
    })

// Add the polylines between chart and labels:
rewardRFPieTextSelect
  .data(rewardRFPieData)
  .enter()
  .append('text')
  .merge(rewardRFPieTextSelect)
    .text( function(d) {  return `#${d.data.key}` } )
    .attr('fill', function(d, i){if(i === cAction)return 'red'; return "white"; return (myColor(d.data.key)) })
    .attr('transform', function(d) {
        var pos = outerArc.centroid(d);
        // var midangle = d.startAngle + (d.endAngle - d.startAngle) / 2
        // pos[0] = radius * 0.99 * (midangle < Math.PI ? 1 : -1);
        pos[0] += 5;
        return 'translate(' + pos + ')';
    })
    .style('text-anchor', function(d) {
        var midangle = d.startAngle + (d.endAngle - d.startAngle) / 2
        return (midangle < Math.PI ? 'start' : 'end')
    })
    .style('text-decoration', 'underline')
    .style('font-size', '20px')
    .style('font-weight', 'bold')
    .style('stroke', 'none')


}

function resetRewardRFPieViz(){

    let rewardRFPiePathSelect = rewardRFPieGroup.selectAll("path");
    let rewardRFPiePolyLineSelect = rewardRFPieGroup.selectAll("polyline");
    let rewardRFPieTextSelect = rewardRFPieGroup.selectAll("text");


    // console.log(rewardRFPieData, rewardRFScaleX(4), rewardRFScaleY(5))
    rewardRFPiePathSelect.data(rewardRFPieData).enter()
    .append('path').merge(rewardRFPiePathSelect)
    .attr('d', arc
    )
    .attr('stroke', 'none')
    .style("stroke-width", "3px")


// Add the polylines between chart and labels:
rewardRFPiePolyLineSelect
  .data(rewardRFPieData)
  .enter()
  .append('polyline')
  .merge(rewardRFPiePolyLineSelect)
    .attr("stroke", "white")
    .style("fill", "none")
    .attr("stroke-width", 1.5)
    .attr('points', function(d) {
      var posA = arc.centroid(d) // line insertion in the slice
      var posB = outerArc.centroid(d) // line break: we use the other arc generator that has been built only for that
      var posC = outerArc.centroid(d); // Label position = almost the same as posB
      var midangle = d.startAngle + (d.endAngle - d.startAngle) / 2 // we need the angle to see if the X position will be at the extreme right or extreme left
      // posC[0] = radius * 0.95 * (midangle < Math.PI ? 1 : -1); // multiply by 1 or -1 to put it on the right or on the left
      return [posA, posB, posC]
    })

// Add the polylines between chart and labels:
rewardRFPieTextSelect
  .data(rewardRFPieData)
  .enter()
  .append('text')
  .merge(rewardRFPieTextSelect)
    .attr('fill', 'white')
    // .attr('transform', function(d) {
    //     var pos = outerArc.centroid(d);
    //     // var midangle = d.startAngle + (d.endAngle - d.startAngle) / 2
    //     // pos[0] = radius * 0.99 * (midangle < Math.PI ? 1 : -1);
    //     pos[0] += 5;
    //     return 'translate(' + pos + ')';
    // })
    // .style('text-anchor', function(d) {
    //     var midangle = d.startAngle + (d.endAngle - d.startAngle) / 2
    //     return (midangle < Math.PI ? 'start' : 'end')
    // })
    // .style('text-decoration', 'underline')
    // .style('font-size', '20px')
    // .style('font-weight', 'bold')
    // .style('stroke', 'none')


}