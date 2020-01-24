
// setting up the svg container
const width = height = 400;

const allMargin = 20;

const pad = 20;


const x = d3.scaleLinear().domain([-5, 5]).range([0,width]);
const y = d3.scaleLinear().domain([ 0, 1]).range([height,0]);

const svg = d3.select("#fxAndDx")
.append("svg")
.attr("width", width + pad + pad)
.attr("height", height + pad + pad)


const legend = d3.select("body").append("div")
.classed("legend", true)

const func = (x) => x**4
