var treeData = [
    {
      "name": "Top Level",
      "parent": "null",
      "children": [
        {
          "name": "Level 2: A",
          "parent": "Top Level",
          "children": [
            {
              "name": "Son of A",
              "parent": "Level 2: A"
            },
            {
              "name": "Daughter of A",
              "parent": "Level 2: A"
            }
          ]
        },
        {
          "name": "Level 2: B",
          "parent": "Top Level"
        }
      ]
    }
  ];

var margin = {top: 80, right: 120, bottom: 20, left: 0},
	width = 360 - margin.right - margin.left,
	height = 500 - margin.top - margin.bottom;
	
var svg = d3.select("#vizMid").append("svg")
	.attr("width", width + margin.right + margin.left)
	.attr("height", height + margin.top + margin.bottom)
	.attr("transform", "translate(" + -220 + "," + 0 + ")")
  .append("g")
	// .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
	.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    let con1Group = svg.append('g').attr('class', 'con1');
    let con2Group = svg.append('g').attr('class', 'con2');

var dataA = {
  source: {
    x: 00,
    y: 00
  },
  target: {
    x: 50,
    y: 50
  }
};

const myD = [];
for(let i=0;i< nArms;i++){

    myD.push( {
       source: {
            x: 210,
            y: 35 + 29*i,
        },
        target: {
            x: 270,
            y: 60 + 33*i ,
        }

    });
}

const myD2 = [];
for(let i=0;i< nArms;i++){

    myD2.push( {
       source: {
            x: -00,
            y: 230,
        },
        target: {
            x: 80,
            y: 35 + 29*i ,
        }

    });
}

var link = d3.linkHorizontal()
  .x(function(d) {
    return d.x;
  })
  .y(function(d) {
    return d.y;
  });

con1Group.selectAll("path").data(myD2).enter().append("path")
  .attr("d", (d,i)=>link(d))
  .style("fill", "none")
  .style("stroke", 'none')
// .style('stroke', 'violet')
  .style("stroke-width", "5px");

con2Group.selectAll("path").data(myD).enter().append("path")
  .attr("d", (d,i)=>link(d))
  .style("fill", "none")
  .style("stroke",(_,i)=>myColor(allMeans[i]))
  .style("stroke", 'none')
// .style('stroke', 'violet')
  .style("stroke-width", "5px");


//Get the total length of the path
var totalLength = 100;

/////// Create the required stroke-dasharray to animate a dashed pattern ///////

//Create a (random) dash pattern
//The first number specifies the length of the visible part, the dash
//The second number specifies the length of the invisible part
var dashing = "6, 6"

//This returns the length of adding all of the numbers in dashing
//(the length of one pattern in essence)
//So for "6,6", for example, that would return 6+6 = 12
var dashLength =
    dashing
        .split(/[\s,]/)
        .map(function (a) { return parseFloat(a) || 0 })
        .reduce(function (a, b) { return a + b });

//How many of these dash patterns will fit inside the entire path?
var dashCount = Math.ceil( totalLength / dashLength );

//Create an array that holds the pattern as often
//so it will fill the entire path
var newDashes = new Array(dashCount).join( dashing + " " );
//Then add one more dash pattern, namely with a visible part
//of length 0 (so nothing) and a white part
//that is the same length as the entire path
var dashArray = newDashes + " 0, " + totalLength;

/////// END ///////

//Now offset the entire dash pattern, so only the last white section is
// //visible and then decrease this offset in a transition to show the dashes
path
    .attr("stroke-dashoffset", totalLength)
    //This is where it differs with the solid line example
    .attr("stroke-dasharray", dashArray)
    .transition().duration(3000).ease("linear")
    .attr("stroke-dashoffset", 0);




function updateInterConnectionViz(cAction, highlightInterval=1000){


  

  con1PathSelect = con1Group.selectAll('path');
  con2PathSelect = con2Group.selectAll('path');

  con1PathSelect.data(myD2).enter().append("path")
  .merge(con1PathSelect)
    .attr("d", (d,i)=>link(d))
    .style("fill", "none")
    .style("stroke",(_,i)=>{return (i===cAction)? myColor(allMeans[i]) : 'none'})
  // .style('stroke', 'violet')
    .style("stroke-width", "10px")
    .attr("stroke-dasharray", "10 10")
    .attr("stroke-dashoffset", "0")
    .transition().duration(highlightInterval/2)
    .attr("stroke-dashoffset", 100);

  con2PathSelect.data(myD).enter().append("path")
  .merge(con2PathSelect)
    .attr("d", (d,i)=>link(d))
    .style("stroke-width", "10px")
    .attr("stroke-dasharray", "10 10")
    .attr("stroke-dashoffset", "0")
    .transition().duration(highlightInterval/2)
    .attr("stroke-dashoffset", 100)
    .style("fill", "none")
    
    .style("stroke",(_,i)=>{return (i===cAction)? myColor(allMeans[i]) : 'none'})
  // // .style('stroke', 'violet')

  setTimeout(
    ()=>{

    con1PathSelect.data(myD2).enter().append("path")
    .merge(con1PathSelect)
      .attr("stroke-dasharray", "")
      .attr("stroke-dashoffset", "0");

    con2PathSelect.data(myD).enter().append("path")
    .merge(con2PathSelect)
      .attr("stroke-dasharray", "")
      .attr("stroke-dashoffset", "0");
    
    }
  ),highlightInterval/2
  }


function resetInterConnectionViz(){

  con1PathSelect = con1Group.selectAll('path');
  con2PathSelect = con2Group.selectAll('path');

  con1PathSelect.data(myD2).enter().append("path")
  .merge(con1PathSelect)
    .attr("d", (d,i)=>link(d))
    .style("fill", "none")
    .style("stroke", 'none')
    .style("stroke-width", "10px");

  con2PathSelect.data(myD).enter().append("path")
  .merge(con2PathSelect)
    .attr("d", (d,i)=>link(d))
    .style("stroke-width", "10px")
    .style("fill", "none")
    .style("stroke", 'none');
}
