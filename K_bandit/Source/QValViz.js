const qValVizElem = document.getElementById('QvalViz');

var margin = {top: 0, right: 30, bottom: 20, left:20},
    width = 400 - margin.left - margin.right,
    height = 50 - margin.top - margin.bottom;

// append the svg object to the body of the page
let Asvg = d3.select("#QvalViz")
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

let qValContainerGroup = Asvg.append('g').attr('id', 'qValContainer');

qValContainerGroup
    .attr("transform", "translate(10," + 0 + ")")

let a = Array(nArms).fill(5);
  // Add X axis
  var qValScaleX = d3.scaleLinear()
    .domain([1, nArms])
    .range([ 0, width ]);

  // Create a Y scale for densities
  var qValScaleY = d3.scaleLinear()
    .domain([0, 1])
    .range([ height, 1]);

  Asvg.append("g")
    .attr("transform", "translate(10," + qValScaleY(0) + ")")
    .call(d3.axisBottom(qValScaleX))
    .selectAll("text")
    //   .attr("transform", "translate(-10,0)rotate(-45)")
      .style("text-anchor", "end");

    qValContainerGroup.selectAll('rect')
    .data(a)
    .enter()
        .append('rect')
        .attr('x', (d,i)=>{ return qValScaleX(i+.55)})
        .attr('y', qValScaleY(0.9))
        .attr('width', (width/(nArms-1)))
        .attr('height', qValScaleY(1-0.9))
        .attr('rx', 2)
        .attr('ry', 2)
        .attr('fill', (d, i)=>{ return myColor(allMeans[i])})
        .attr('stroke-width', 2)
        .attr('stroke', 'none');

    qValContainerGroup.selectAll('text')
    .data(a)
    .enter()
        .append('text')
        .attr('text-anchor', 'end')
        .attr('x', (d,i)=>{
            return qValScaleX(i+.3+1)})
        .attr('y', qValScaleY(0.1))
        .text((d,i) =>{return d})
        .attr('fill', 'white')



//   rewardRFSvg.append("text")
//       .attr("text-anchor", "end")
//       .attr("x", width)
//       .attr("y", height+20)
//       .text('N(a)')

function updateQVal(action, rVals){
    console.log('alsdkjflkasjdf', rVals)
    const textSelect = qValContainerGroup.selectAll('text')
    const rectSelect = qValContainerGroup.selectAll('rect')

    textSelect.remove();

    rectSelect.data(rVals)
    .enter().append('rect')
    .merge(rectSelect)
    .attr('stroke', (d,i)=>{return (i === action)? 'red': 'none'})

   qValContainerGroup.selectAll('text') 
    .data(rVals)
    .enter()

        .append('text')
        .attr('text-anchor', 'end')
        .attr('x', (d,i)=>{
            return qValScaleX(i+.3+1)})
        .attr('y', qValScaleY(0.1))
        .text((d,i) =>{return d.toFixed(1)})
        .attr('fill', 'white')

}