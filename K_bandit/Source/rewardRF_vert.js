// set the dimensions and margins of the graph
var margin = {top: 20, right: 30, bottom: 40, left: 50},
    width = 200 - margin.left - margin.right,
    height = 400 - margin.top - margin.bottom;

// append the svg object to the body of the page
var rewardRFSvg = d3.select("#rewardRFViz")
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");


let rewardRFBarGroup = rewardRFSvg.append('g').attr('class', 'relFreqBars');

// contain normalized relative frequency...
let rewardRFArray = Array(nArms).fill(0);

// for(let i=0;i<nArms;i++){
//     rewardRFArray[i] = Math.random()
// }

// rewardRFArray = normalize(rewardRFArray);


let rewardRFData = rewardRFArray.map((d,i)=>{ return {x: d,y:i}})

  // Add X axis
  var rewardRFScaleX = d3.scaleLinear()
    .domain([0, 1])
    .range([ 0, width]);
  rewardRFSvg.append("g")
    .attr("transform", "translate(0," + height + ")")
    // .call(d3.axisBottom(rewardRFScaleX))
    .selectAll("text")
      .attr("transform", "translate(-10,0)rotate(-45)")
      .style("text-anchor", "end");

  // Y axis
  var rewardRFScaleY = d3.scaleBand()
    .range([ 0, height ])
    .domain([0, 1,2,3,4,5,6,7,8,9])
    .padding(.1);
  rewardRFSvg.append("g")
    .call(d3.axisLeft(rewardRFScaleY))

  //Bars
    rewardRFBarGroup.selectAll('myRect')
    .data(rewardRFData)
    .enter()
    .append("rect")
    .attr("x", rewardRFScaleX(0) )
    .attr("y", function(d) { return rewardRFScaleY(d.y); })
    .attr("width", function(d) { return rewardRFScaleX(d.x); })
    .attr("height", rewardRFScaleY.bandwidth() )
    .attr("fill", "#69b3a2")


  rewardRFSvg.append("text")
      .attr("text-anchor", "end")
      .attr("x", width)
      .attr("y", height+20)
      .text('N(a)')

    // .attr("x", function(d) { return x(d.Country); })
    // .attr("y", function(d) { return y(d.Value); })
    // .attr("width", x.bandwidth())
    // .attr("height", function(d) { return height - y(d.Value); })
    // .attr("fill", "#69b3a2")



function normalize(array){

    const total = d3.sum(array);
    return array.map((v)=>{return v/total})
}


function updateRewardRF(cAction,array){

    array = normalize(array);
    let rewardRFData = array.map((d,i)=>{ return {x: d,y:i}});

    let rewardRFRectSelect = rewardRFBarGroup.selectAll("rect");

    // console.log(rewardRFData, rewardRFScaleX(4), rewardRFScaleY(5))
    rewardRFRectSelect.data(rewardRFData).enter()
    .append('rect').merge(rewardRFRectSelect)
    .attr("x", rewardRFScaleX(0) )
    .attr("y", function(d) { return rewardRFScaleY(d.y); })
    .attr("width", function(d) { return rewardRFScaleX(d.x); })
    .attr("height", rewardRFScaleY.bandwidth() )
    .attr("fill", (d,i)=>{return( (i === cAction)? "red" : "#69b3a2" )});

}
