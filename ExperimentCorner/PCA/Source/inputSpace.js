// synthetic data
let data0 = tf.randomNormal([1,10],0,1).transpose();
let data1 = tf.randomNormal([1,10],0,1).transpose();


let data = data0.concat(data1,axis=1).arraySync();

// PCA({x: data ,y:[]})


data =([[ -6.84620013,  -6.69294004],
  [  4.33280112,   5.02483149],
  [ -0.96618573,  -2.86864073],
  [ -1.00394315,  -0.62668941],
  [  1.51062592,   0.58568872],
  [  2.39966693,   0.76864898],
  [-10.24696564,  -6.32731168],
  [  7.32302626,   5.82945762],
  [  2.03822847,   0.39973107],
  [  1.45894596,   3.90722399]])




// const ourDataPoints = [];


// Helper Function:-

function click(){
  // Ignore the click event if it was suppressed
  // if (d3.event.defaultPrevented) return;

  // Extract the click location\    
  var point = d3.mouse(this)
  , p = {x: point[0], y: point[1] };

  // PCA({x: ourDataPoints, y:[]})


console.log('yes')
// console.log(svg.selectAll('circle'))
  // Append a new point
  svg.select('.dataPoints').append("circle")
      .attr('cx', (p.x -margin.left))
      .attr('cy', (p.y -margin.top))
      .attr("r", "5")
      .style("fill", "#69b3a2")
      .call(drag)
      
}

// Define drag beavior
var drag = d3.drag()
    .on("drag", dragmove);

function dragmove(d) {
  var x = d3.event.x;
  var y = d3.event.y;
  d3.select(this)
      .attr('cx', (x ))
      .attr('cy', (y ))
}




// Visualization:-

// range of the plot
const range = {min: -2, max: 10};

// set the dimensions and margins of the graph
var margin = {top: 10, right: 30, bottom: 30, left: 60},
    width = 460 - margin.left - margin.right,
    height = 400 - margin.top - margin.bottom;

// append the svg object to the body of the page
var svg = d3.select("#dataInput")
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .on("click", click)
    .call(d3.zoom().on("zoom", function () {
            svg.attr("transform", d3.event.transform)
    }))
.append("g")
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")")


// Add X axis
let x = d3.scaleLinear()
  .domain([range.min, range.max])
  .range([ 0, width ]);
svg.append("g")
  .attr("transform", "translate(0," + height + ")")
  .call(d3.axisBottom(x));


// Add Y axis
let y = d3.scaleLinear()
  .domain([range.min, range.max])
  .range([ height, 0]);
svg.append("g")
  .call(d3.axisLeft(y));

// converting the pixel coordinate back to the desired range
let xInv = d3.scaleLinear()
  .domain([ 0, width ])
  .range([range.min, range.max]);

let yInv = d3.scaleLinear()
  .domain([ height, 0])
  .range([range.min, range.max]);

let gridIntervels = tf.linspace(range.min+1, range.max-1, 8).flatten().arraySync();

var lineGenerator = d3.line();

// vertical grid lines 
svg.append('g')
  .selectAll('path')
  .data(gridIntervels)
  .enter()
  .append('path')
  .attr( 'd', function(d) {return lineGenerator([[x(d),y(range.min)],[x(d),y(range.max)]])} )
  .style('stroke', "gray")
  .attr('opacity', 0.5)

// vertical grid lines 
svg.append('g')
  .selectAll('path')
  .data(gridIntervels)
  .enter()
  .append('path')
  .attr( 'd', function(d) {return lineGenerator([[x(range.min),y(d)],[x(range.max),y(d)]])} )
  .style('stroke', "gray")
  .attr('opacity', 0.5)


const originAxis = svg.append('g').attr('class', 'originAxis')

// origin x-axis

originAxis
  .append('path')
  .attr( 'd', lineGenerator([[x(range.min),y(0)],[x(range.max),y(0)]]) )
  .attr('stroke-width', 3)
  .attr('opacity', 0.5)
  .attr('stroke', 'blue');

// origin y-axis
originAxis
  .append('path')
  .attr( 'd', lineGenerator([[x(0),y(range.min)],[x(0),y(range.max)]]) )
  .attr('stroke-width', 3)
  .attr('opacity', 0.5)
  .attr('stroke', 'red');


// Plotting the synthetic data...
// svg.select('.dataPoints')
//   .selectAll("dot")
//   .data(data)
//   .enter()
//   .append("circle")
//     .attr("cx", function (d) { return x(d[0]); } )
//     .attr("cy", function (d) { return y(d[1]); } )
//     .attr("r", 5)
//     .style("fill", "#69b3a2")


svg.append('g')
  .attr('class', 'dataPoints')