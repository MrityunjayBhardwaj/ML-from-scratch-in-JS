// Currently, creating a gif of 5 component GMM in 3d with rotation

let d3 = Plotly.d3;
let img_jpg = d3.select('#jpg-export');
const gd= document.getElementById('GMMViz');

// initializing data
let mIrisX = tf.tensor(iris).slice([0,1],[150,2])
mIrisX = normalizeData(mIrisX,unitVariance=1);
// one hot encoded
let mIrisY = Array(150).fill([1,0],0,50).fill([0,1],50);

let augIrisData = mIrisX.concat(tf.tensor(mIrisY),axis=1).arraySync();
tf.util.shuffle(augIrisData,axis=0);
augIrisData = tf.tensor( augIrisData );

let testX =   augIrisData.slice([0,0],[25,2]);
let testY =   augIrisData.slice([0,2],[25,-1]);

const mIrisXArray = mIrisX.arraySync();
tf.util.shuffle(mIrisXArray);
mIrisX = tf.tensor(mIrisXArray);



const k = 5;

const myMeans = JSON.parse(meanString); 
const myCovariance = JSON.parse(covarianceString);
const myMixingCoeff = JSON.parse(mixingCoeffString);

let mean = [];
let covariance = [];
let mixingCoeff =  [];




let np = 0; 


let forReplacing = "(";
let myModel =0;

function pGif(p){
    mean = [];
    covariance = [];
    mixingCoeff =  [];

    const currTimeStep = np;

    for(let i=0;i<k;i++){

        mean.push(tf.tensor(myMeans[currTimeStep][i]))
        covariance.push(tf.tensor(myCovariance[currTimeStep][i]))
        mixingCoeff.push(tf.tensor(myMixingCoeff[currTimeStep][i]))

    }

    myModel = new GMM(mean, covariance, mixingCoeff);

    const gridRange0 = [-2, 2];
    const gridRange1 = [-3, 3.5];
    const psudoPtsGridRes = 100;

    const psudoPts0 = tf.linspace( gridRange0[0],gridRange0[1],psudoPtsGridRes).expandDims(1);
    const psudoPts1 = tf.linspace( gridRange1[0],gridRange1[1],psudoPtsGridRes).expandDims(1);
    const psudoPts  = psudoPts0.concat(psudoPts1,axis=1);

    let meshGridPsudoPts = tf.tensor(meshGrid(psudoPts1.flatten().arraySync(),psudoPts0.flatten().arraySync()))
    meshGridPsudoPts = meshGridPsudoPts.reshape([meshGridPsudoPts.shape[0]*meshGridPsudoPts.shape[1], meshGridPsudoPts.shape[2]]);
    let meshGridZ = myModel.test(meshGridPsudoPts)

    console.log(meshGridZ.print())

    const angle = Math.PI*2*(np/50);
    const layoutSetting = {
        title: 'Gaussian Mixture Model',
        font : {
            size : 15,
            color: 'white',
            family : 'Helvetica'
        },
        paper_bgcolor : '#222633',
        width: 800,
        height: 800,
        scene: {
            camera: {
                // eye: {
                //     x: 1.0,
                //     y: 1.50,
                //     z: 1.00,
                // },
                eye: rotate('scene', {x: 1.0,y:1.50,z: 1.0}, angle)

                // up: {
                //     x: 1*Math.sin(angle), 
                //     y: 0*Math.cos(angle),
                //     z: 1*Math.sin(angle),
                // },

                // center: {
                //     x: 0,
                //     y: 0,
                //     z: 0,
                // }
            }
        }

    }

    if(np !== 0)
        rotate('scene',angle*Math.PI/180)

    const clusterData = [

            // {
            //     x: mIrisX.slice([0,0],[-1,1]).flatten().arraySync(),
            //     y: mIrisX.slice([0,1],[-1,-1]).flatten().arraySync(),
            //     z: (new Array(mIrisX.shape[0])).fill(0),
            //     type: 'scatter',
            //     // type: 'scatter3d',

            //     mode: 'markers',
            //     name: ' data',

            //     marker : {
            //         size: 10,
            //     }
            // },

            {  
                // x: meshGridPsudoPts.slice([0,0],[-1,1]).flatten().arraySync(),
                // y: meshGridPsudoPts.slice([0,1],[-1,1]).flatten().arraySync(),
                // z: meshGridZ.flatten().arraySync(),

                x: psudoPts0.flatten().arraySync(),
                y: psudoPts1.flatten().arraySync(),
                z: meshGridZ.reshape([meshGridZ.shape[0]**(1/2), meshGridZ.shape[0]**(1/2)]).arraySync(),

                // intensity: meshGridZ.flatten().arraySync(),
                // type: "mesh3d",
                // type: 'contour',
                type: 'surface',
                // opacity : 0.7,
                // color: 'pink'

                colorscale: [
                    [0, darkModeCols.blue()],
                    [0.25, darkModeCols.purple()],
                    [0.5, darkModeCols.magenta()],
                    [0.75, darkModeCols.yellow()],
                    [1, darkModeCols.red()]
                ],
            }



    ];


    Plotly.newPlot('GMMViz', clusterData, layoutSetting)
    .then(
            function(gd)
            {
            Plotly.toImage(gd,{height:800,width:800})
                .then(
                    function(url2)
                {
                    console.log(url2);  
                    img_jpg.attr("src", url2);

                let img = document.images[0];
                    // atob to base64_decode the data-URI
                    let image_data = atob(img.src.split(',')[1]);
                    // Use typed arrays to convert the binary data to a Blob
                    let arraybuffer = new ArrayBuffer(image_data.length);
                    let view = new Uint8Array(arraybuffer);
                    let blob = 0;
                    for (let i=0; i<image_data.length; i++) {
                        view[i] = image_data.charCodeAt(i) & 0xff;
                    }
                    try {
                        // This is the recommended method:
                        blob = new Blob([arraybuffer], {type: 'application/octet-stream'});
                    } catch (e) {
                        // The BlobBuilder API has been deprecated in favour of Blob, but older
                        // browsers don't know about the Blob constructor
                        // IE10 also supports BlobBuilder, but since the `Blob` constructor
                        //  also works, there's no need to add `MSBlobBuilder`.
                        let bb = new (window.WebKitBlobBuilder || window.MozBlobBuilder);
                        bb.append(arraybuffer);
                        let blob = bb.getBlob('application/octet-stream'); // <-- Here's the Blob
                    }

                    // Use the URL object to create a temporary URL
                    let url = (window.webkitURL || window.URL).createObjectURL(blob);
                    location.href = url; // <-- Download!

                    forReplacing = forReplacing.concat(" ["+url.slice(13)+"]="+np+".png")

                    console.log("url: ",url, forReplacing)

                    myModel.test(tf.tensor([[.8,-1.3],])).print()


                        np++;
                        if(np < 200){
                            pGif(np);
                            sleep(1000);

                            return Plotly.toImage(gd,{format:'jpeg',height:800,width:800});
                        }
                        else{

                            forReplacing = forReplacing.concat(" )")
                        }
                        }
                        )
                    });



}

pGif(np);




function rotate(id,eye0, angle) {
    // var eye0 = gd.layout[id].camera.eye
    var rtz = xyz2rtz(eye0);
    rtz.t += angle;
    
    var eye1 = rtz2xyz(rtz);

    console.log(eye1)

    return eye1;
    Plotly.relayout(gd, id + '.camera.eye', eye1)
  }
  
  function xyz2rtz(xyz) {
    return {
      r: Math.sqrt(xyz.x * xyz.x + xyz.y * xyz.y),
      t: Math.atan2(xyz.y, xyz.x),
      z: xyz.z
    };
  }
  
  function rtz2xyz(rtz) {
    return {
      x: rtz.r * Math.cos(rtz.t),
      y: rtz.r * Math.sin(rtz.t),
      z: rtz.z
    };
  }