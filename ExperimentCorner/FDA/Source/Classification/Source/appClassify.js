  // initializing data
  const mIrisX = tf.tensor(iris).slice([0,0],[100,2])
  // one hot encoded
  const mIrisY =  Array(100).fill([1,0],0,50).fill([0,1],50) ;
  /**
   * 
   * @param {number} x cursor location X
   * @param {number} y cursor location Y
   * @param {object} originalRange of x and y-axis { x : [min,max], y: [min,max] }
   * @param {object} newRange range of x and y-axis { x : [min,max], y: [min,max] }
   * @description this function just maps the x and y coordinate value to newRange Space given the range of the domain of x and y values. 
   * @returns returns the coordinate points according to the parameters.
   */
  function reMapCoods(x,y,originalRange,newRange){

    const coordX = remap(x,originalRange.x[0],originalRange.x[1],newRange.x[0],newRange.x[1]);
    const coordY = remap(y,originalRange.y[0],originalRange.y[1],newRange.y[0],newRange.y[1]);

    return [coordX,coordY];
  }

  const data0 = {x: [0,5], y : [0, 5]};
  const data1 = {x: [0,5], y : [0, 5]};
  let outputData = [
    {},{},{}
  ];

  
  const fac = 2;// projection vector scaling factor

  let projVec = [0,0];

  var traces = [{
    name: "Class-A Data",
    x: data0.x,
    y: data0.y,
    mode: 'markers',
    type: 'scatter'
  },
  {
    name: "Class-B Data",
    x: data1.x,
    y: data1.y,
    mode: 'markers',
    type: 'scatter'
  },{
      name: "FDA's Projection Vector",
      x : [ -projVec[0]*fac ,projVec[0]*fac ],
      y : [ -projVec[1]*fac ,projVec[1]*fac ],
      mode: 'lines',
      type: 'scatter',
      line : { width : 4}
    },
    {}
  ];

  let layout = {
    
    title: "Input Space",
    showlegend : false,
    xaxis: {
      range: [-1,5],
      autorange: false 
    },
    yaxis: {
      range: [-1,5],
      autorange: false 
    },    
    margin: {
      autoexpand: false,
    },
  };

  let outlayout = {
    
    title: "Output Space",
    showlegend : false,
    xaxis: {
      range: [-1,5],
      autorange: false 
    },
    yaxis: {
      range: [-1,5],
      autorange: false 
    },
    margin: {
      autoexpand: false,
    },
  };
  var myPlot = document.getElementById('intractiveInput')
  Plotly.newPlot('interactiveInput', traces, layout,{staticPlot: true});
 

  // for output Region
  Plotly.newPlot('outputViz',outputData, layout ,{staticPlot: true});

  Number.prototype.between = function(min, max) {
    return this >= min && this <= max;
  };

  const toggle =  document.getElementById('myCheck');
  let selClass = toggle.checked;

  console.log(selClass);

  document.addEventListener('keydown',function(event){
    if (event.key == 1){
      toggle.checked = false ;
    }
    if (event.key == 2){
      toggle.checked = true ;
    }
    //  console.log(event.key == 2)
    });

  // document.getElementsByClassName('svg-container')[1]
  Plotly.d3.select(".plotly").on('click', function(d, i) {
    var e = Plotly.d3.event;
    // var bg = document.getElementsByClassName('svg-container')[1];
    let plotContainer = document.getElementsByClassName('svg-container')[1];

    const divCoord = {x: e.layerX, y: e.layerY};
    const margin = 80;
    // console.log(plotContainer.offsetWidth,100,divCoord)
    if( divCoord.x.between( margin, plotContainer.offsetWidth - margin) &&
        divCoord.y.between( margin, plotContainer.offsetHeight - margin ) ){
          const graphPix = {x: divCoord.x -margin*1 , y : (plotContainer.offsetHeight - divCoord.y) - margin*1}

          const offset = 0;
          const cvsRange = { x : [margin*0, plotContainer.offsetWidth -margin*2 -offset ] , y: [margin*0, plotContainer.offsetHeight - margin*2 - offset]} ;
          const graphCoords = reMapCoods(graphPix.x,graphPix.y,cvsRange, {x : [-1,5],y: [-1,5]});

          selClass = document.getElementById('myCheck').checked;
          console.log(selClass*1,graphPix);

          const classData = traces[selClass*1];
            
          classData.x.push(graphCoords[0]);
          classData.y.push(graphCoords[1]);

          console.log(graphCoords);

          if ( (traces[0].x.length > 3) && (traces[1].x.length > 3) ){

            // initializing data
            let dataC0 = tf.tensor( [ traces[0].x.slice(2,), traces[0].y.slice(2,) ]).transpose();
            let dataC1 = tf.tensor( [ traces[1].x.slice(2,), traces[1].y.slice(2,) ]).transpose();

            let dataX = dataC0.concat(dataC1, axis=0);
            let dataY = Array(dataC0.shape[0] + dataC1.shape[0]).fill([1,0],0,dataC0.shape[0]).fill([0,1],dataC0.shape[0],);

            const model = new FDAmc();
            projVec = model.train({ x:dataX.arraySync(), y:dataY });

            
            // const projDataX = tf.matMul(dataX,projVec);
              console.log(projVec);
               
            // const orth2EigVec = nd
            // const bias = - projDataX;

            const projVecViz = [{
              name: "FDA's Projection Vector",
              x : [ -projVec[0][0]*fac ,projVec[0][0]*fac ],
              y : [ -projVec[0][1]*fac ,projVec[0][1]*fac ],
              mode : 'lines',
              type : 'scatter',
              line : { width : 4, color: 'green'}
            },
            {
              name: "FDA's Projection Vector",
              x : [ -projVec[1][0]*fac ,projVec[1][0]*fac ],
              y : [ -projVec[1][1]*fac ,projVec[1][1]*fac ],
              mode : 'lines',
              type : 'scatter',
              line : { width : 4, color: 'green'}
            }
          
          ]
            traces[2] = projVecViz[0];
            traces[3] = projVecViz[1];

            // TODO: figure out the decision boundary
            // let decisionBoundary = tf.linalg.gramSchmidt(tf.tensor(projVec).expandDims(1).transpose()).flatten().arraySync();
            // const decBoundaryViz = {
            //   name: "Decision Boundary ",
            //   x : [ -decisionBoundary[0]*fac ,decisionBoundary[0]*fac ],
            //   y : [ -decisionBoundary[1]*fac ,decisionBoundary[1]*fac ],
            //   mode: 'lines',
            //   type: 'scatter',
            //   line : { width : 4}
            // }
            // traces[3] = (decBoundaryViz);

            const psudoPtsGridRes = 100;
            const psudoPts0 = tf.linspace( -1, 5, psudoPtsGridRes );
            const psudoPts  = tf.tile( psudoPts0.expandDims(1).transpose(),[2,1] ).transpose(); // mashGrid
            const psudoPty  = model.transform( psudoPts );
            const psudoPtsClassify = model.classify(psudoPts)
            window.model  = model;

            console.log( psudoPts );
            const meshGridPsudoPts = meshGrid(psudoPts0.arraySync(),psudoPts0.arraySync());
            const meshGridPsudoPtsY = meshGridPsudoPts.map( 
                  function(cRow) {
                    return model.classify( tf.tensor(cRow) ).flatten().arraySync()
                  }
             );

            const decisionRegionData = {
              x : psudoPts.slice([0,0],[-1,1]).flatten().arraySync(),
              y : psudoPts.slice([0,1],[-1,1]).flatten().arraySync(),
              z : meshGridPsudoPtsY,
              type: 'contour',
                 colorscale:[[0, 'rgb(153, 153, 255)'],[1, 'rgb(255, 153, 102)']],
              contours : {
                // coloring : 'heatmap',
                // zsmooth: 'best',
              },
              line : {
                width: 0,
                 smoothing: 0.85
              },

            }
            console.log(dataX.print())

            outputData[0] = { 
              x : dataX.slice([0,0],[-1,1]).flatten().arraySync(), 
              y: dataX.slice([0,1],[-1,-1]).flatten().arraySync(),
              mode: 'markers',
              type: 'scatter',
              marker : {
                width : 5,
                //  symbol: ["diamond-open"],
                 color: 'gray'
              }
            }
            outputData[2] = decisionRegionData;
            // traces[3] = {
            //   x: 
            // }

            plotDist(dataX.arraySync(),dataY,'2classDistViz',0);
            plotDist(dataX.arraySync(),dataY,'2classDistViz',1);

          }

          // updating the plot
          Plotly.newPlot('interactiveInput', traces, layout ,{staticPlot: true});
          Plotly.newPlot('outputViz', outputData, outlayout ,{staticPlot: true});

          // TODO:  FIX:  1 unequal sample size and only add when its > 2

        }

    window.dum = e;
    // console.log(e.layerX,e.layerY);
  });

  plotDist(mIrisX.arraySync(),mIrisY,'2classDistViz');





function plotDist(dataX,dataY,containerId,index = 0){


  const tfDataX = tf.tensor(dataX);
  const tfDataY = tf.tensor(dataY);

  // calculating the projection vector
  const model = new FDAmc();
  let projVec = model.train({ x:dataX, y:dataY });
  projVec = tf.tensor(projVec)
  projVec = projVec.slice([0,1],[-1,-1]);

  // ( (new FDA()).train({x : dataX,y: dataY}).print() )
  // const model = new FDA();
  // let projVec = model.train({ x:dataX, y:dataY });


  // projecting the data X
  let clfProjX = tf.matMul(tfDataX,projVec);

  const dataSplit = classwiseDataSplit(clfProjX,tfDataY);

  // split the data
  const clfProjX0 =  dataSplit[0].x;
  const clfProjX1 =  dataSplit[1].x;

  // calculating parameters of gaussian

  // calculating mean
  const clfProjX0mean = tf.mean(clfProjX0).flatten().arraySync()[0];
  const clfProjX1mean = tf.mean(clfProjX1).flatten().arraySync()[0];

  // calculating variance
  const clfProjX0var =  tf.mul( tf.sum( tf.pow( tf.sub(clfProjX0,clfProjX0mean), 2 ) ), 1/clfProjX0.shape[0] ).flatten().arraySync()[0];
  const clfProjX1var =  tf.mul( tf.sum( tf.pow( tf.sub(clfProjX1,clfProjX1mean), 2 ) ), 1/clfProjX1.shape[0] ).flatten().arraySync()[0];

  // using calculated parameter to form a gaussian
  function gauss (x,mu,sigma){ return ( 1/Math.sqrt(2*Math.PI*(sigma**2)) ) * Math.exp( -1/2 * ((x - mu)**2)/sigma**2 ); };

  // using gaussians to calculate the probability of each data point
  // const clfProjX0prob = clfProjX0.flatten().arraySync().map( (x)=> gauss(x,clfProjX0mean,Math.sqrt(clfProjX0var)) );
  // const clfProjX1prob = clfProjX1.flatten().arraySync().map( (x)=> gauss(x,clfProjX1mean,Math.sqrt(clfProjX1var)) );

  const gauss0 = new NormalDistribution(clfProjX0mean, 1*Math.sqrt(clfProjX0var));
  const gauss1 = new NormalDistribution(clfProjX1mean, 1*Math.sqrt(clfProjX1var));

  // console.log(clfProjX0var,clfProjX1var)
  // parameters for generationg psudo Data
  let margin0 = Math.abs(tf.min(clfProjX0).arraySync() - tf.max(clfProjX0).arraySync()) *(20/clfProjX0.shape[0]);
  let margin1 = Math.abs(tf.min(clfProjX1).arraySync() - tf.max(clfProjX1).arraySync()) *(20/clfProjX1.shape[0]);

  margin0 = Math.sqrt(clfProjX0var)*3;
  margin1 = Math.sqrt(clfProjX1var)*3;
  const resolution = 100;

  // creating psudoData
  const psudoX0 = tf.linspace(tf.min(clfProjX0).flatten().arraySync()[0]-margin0,tf.max(clfProjX0).flatten().arraySync()[0]+margin0,resolution);
  const psudoX1 = tf.linspace(tf.min(clfProjX1).flatten().arraySync()[0]-margin1,tf.max(clfProjX1).flatten().arraySync()[0]+margin1,resolution);

  // calculating the probability of psudo data
  const psudoX0prob = psudoX0.flatten().arraySync().map( function(x) {return gauss0.pdf(x)});
  const psudoX1prob = psudoX1.flatten().arraySync().map( function(x) {return gauss1.pdf(x)});

  // console.log(clfProjX0prob);
  // console.log(clfProjX1prob);

  // plotting projected output
  const clfProjXData = [
      {
          name: "Class-A PDF",
          legendgroup : "Class-A",
          x: psudoX0.flatten().arraySync(), 
          y: psudoX0prob,
          mode: 'lines',
          type: 'scatter',
          line: {width: 4, color: 'blue'},
      },
      {
          name: "Class-B PDF",
          legendgroup : "Class-B",
          x: psudoX1.flatten().arraySync(),
          y: psudoX1prob,
          mode: 'lines',
          type: 'scatter',
          line: {width: 4, color: 'orange',lagend : 'hsdf'},
      },
      {
          name: "Class-A data",
          legendgroup : "Class-A",
          x: clfProjX0.flatten().arraySync(),
          y: Array(50).fill(0),
          mode: 'markers',
          type: 'scatter',
          marker: {width: 2, color: 'blue'}
      },
      {
          name: "Class-B data",
          legendgroup : "Class-B",
          x: clfProjX1.flatten().arraySync(),
          y: Array(50).fill(0),
          mode: 'markers',
          type: 'scatter',
          marker: {width: 2, color: 'orange'}
      },
      
  ];

  const classifierLayout = {
    title: 'FDA for Classification',

    xaxis:{
        range : [-4, 4],
        autorange: false,
    },

    yaxis:{
        autorange: false
    }
    
  }

  Plotly.newPlot(containerId, clfProjXData, classifierLayout)

}