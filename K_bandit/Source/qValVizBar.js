// set the dimensions and margins of the graph
var margin = {top: 20, right: 30, bottom: 40, left: 35},
    qValVizWidth = 470 - margin.left - margin.right,
    qValVizHeight = 200 - margin.top - margin.bottom;

const qValBarVizElem = document.getElementById("qValBarViz");
const viz3Elem = document.getElementById('viz3');

// append the svg object to the body of the page
var qValBarSvg = d3.select(qValBarVizElem)
  .append("svg")
    .attr("width", qValVizWidth + margin.left + margin.right)
    .attr("height", qValVizHeight + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");


let qValBarGroup = qValBarSvg.append('g').attr('class', 'qValBars');
let qValBarAxisLeftGroup = qValBarSvg.append('g').attr('id', 'axisLeft');

let qValBarAxisMidGroup = qValBarSvg.append('g').attr('id', 'axisMid');

const qValBarValueGroup = qValBarSvg.append('g').attr('id', 'qValues');

// contain normalized relative frequency...
let qValBarArray = Array(nArms).fill(0);

// for(let i=0;i<nArms;i++){
//     qValBarArray[i] = Math.random()
// }

// qValBarArray = normalize(qValBarArray);


let qValBarData = qValBarArray.map((d,i)=>{ return {x: .3,y:i}})

  // Add X axis
  var qValBarScaleX = d3.scaleBand()
    .range([ 0, qValVizWidth])
    .domain([0, 1,2,3,4,5,6,7,8,9])
    .padding(.1);
  qValBarSvg.append("g")
    .attr("transform", "translate(0," + qValVizHeight + ")")
    .call(d3.axisBottom(qValBarScaleX))
    .selectAll("text")
      // .attr("transform", "translate(-10,0)rotate(-45)")
      .style("text-anchor", "end");
      

  // Y axis
  var qValBarScaleY = d3.scaleLinear()
    .domain([-1, 1])
    .range([ qValVizHeight, 0 ]);

    qValBarAxisLeftGroup
    .call(d3.axisLeft(qValBarScaleY).ticks(5))
    .selectAll("text")
      // .attr("transform", "translate(-10,0)rotate(-45)")
      .style("text-anchor", "end");
    
    ;
    // .attr('fill', 'white');

  qValBarAxisMidGroup
    // .attr("transform", "translate(0," + qValVizHeight + ")")
    // .selectAll('rect').data(['2'])
    // .enter()
    .append('rect')
    .attr('x', qValBarScaleX(0))
    .attr('y', qValBarScaleY(0))
    .attr('width', qValVizWidth)
    .attr('height', "2px")
    .attr('fill', 'white');

    qValBarValueGroup.selectAll('text')
    .data(qValBarData)
    .enter()
    .append('text')
    .text("5.0")
    .attr('transform', (d,i)=>{return `translate(${qValBarScaleX(i)+margin.left-7}, ${qValBarScaleY(0)})`})
    .style("text-anchor", "end");

  //Bars
    qValBarGroup.selectAll('myRect')
    .data(qValBarData)
    .enter()
    .append("rect")
    .attr("y", (d,i)=>{return qValBarScaleY(d.x)})
    .attr("x", function(d) { return qValBarScaleX(d.y); })
    .attr("height", function(d) { return qValBarScaleY(1-d.x); })
    .attr("width", qValBarScaleX.bandqValVizWidth() )
    .attr('fill', (d, i)=>{ return myColor(allMeans[i])});
    // .attr("fill", "#69b3a2")





function updateQValBar(cAction,array,highlightInterval=1000){

    viz3Elem.style.borderColor="red"; 
    setTimeout(()=>{

        viz3Elem.style.borderColor="";
    }, highlightInterval);

   qValBarScaleY = d3.scaleLinear()
    .domain([d3.min(array)-.6, d3.max(array)+.6])
    .range([ qValVizHeight, 0 ]);

    qValBarAxisLeftGroup
    .call(d3.axisLeft(qValBarScaleY).ticks(5))
    .selectAll("text")
      // .attr("transform", "translate(-10,0)rotate(-45)")
      .style("text-anchor", "end");
    // .ticks(10);

  qValBarAxisMidGroup.select('rect')
    .attr('y', qValBarScaleY(0))





    let qValBarData = array.map((d,i)=>{ return {x:i,y: d}});

    // updating qValues text
    const qValBarValueGroupTextSelect = qValBarValueGroup.selectAll('text')

    qValBarValueGroupTextSelect
    .data(qValBarData)
    .enter()
    .append('text')
    .merge(
    qValBarValueGroupTextSelect
    )
    .text((d,i)=>{return d.y.toFixed(1)})
    .attr('transform', (d,i)=>{return `translate(${qValBarScaleX(i)+margin.left+30}, ${ (d.y >= 0)? qValBarScaleY(d.y*1)-5 :  qValBarScaleY(d.y*1)+15})`})
    .style("fill", (d,i)=>{return( (i === cAction)? "red" : 'whitesmoke')})
    .style('stroke-width', 0)
    .style('font-weight', 'bold')
    .style("text-anchor", "end");
        // .call(d3.axisBottom(qValBarScaleX))
    // array = normalize(array);

    let qValBarRectSelect = qValBarGroup.selectAll("rect");

    qValBarRectSelect
    .data(qValBarData)
    .enter()
    .append("rect").merge(qValBarRectSelect)
    // .attr("y", (d,i)=>{  return (d.y >= 0)?  qValBarScaleY(d.y) : qValBarScaleY(Math.abs(d3.min(array)) - Math.abs(d.y)) })
    .attr("y", (d,i)=>{  return (d.y>=0)? qValBarScaleY(d.y*1) : qValBarScaleY(0)})
    .attr("x", function(d) { return qValBarScaleX(d.x); })
    .attr("height", function(d) { return (d.y >= 0)? qValBarScaleY(0)-qValBarScaleY(d.y) : Math.abs(qValBarScaleY(0)-qValBarScaleY(d.y)); })
    // .attr("height", function(d) { return qValBarScaleY(d.y) })
    .attr("width", qValBarScaleX.bandwidth() )
    .attr("fill", (d,i)=>{return( (i === cAction)? "red" : myColor(allMeans[i]) )});
    // .attr('fill', (d, i)=>{ return myColor(allMeans[i])})

}

function resetQValBarViz(){

    const qValBarValueGroupTextSelect = qValBarValueGroup.selectAll('text')

    qValBarValueGroupTextSelect
    .data(qValBarData)
    .enter()
    .append('text')
    .merge(
    qValBarValueGroupTextSelect
    )
    .style("fill", (d,i)=>{'whitesmoke'})


    let qValBarRectSelect = qValBarGroup.selectAll("rect");

    qValBarRectSelect
    .data(qValBarData)
    .enter()
    .append("rect").merge(qValBarRectSelect)
    .attr("fill", (d,i)=>{return myColor(allMeans[i]) });
}