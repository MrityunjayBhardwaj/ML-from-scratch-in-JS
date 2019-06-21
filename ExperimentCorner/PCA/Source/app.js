
const csvUrl =
'https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/boston-housing-train.csv';


async function run() {
  // We want to predict the column "medv", which represents a median value of
  // a home (in $1000s), so we mark it as a label.
  const csvDataset = tf.data.csv(
    csvUrl, {
      columnConfigs: {
        medv: {
          isLabel: true
        }
      }
    });

  // Number of features is the number of column names minus one for the label
  // column.
  const numOfFeatures = (await csvDataset.columnNames()).length - 1;

  // Prepare the Dataset for training.
  const flattenedDataset =
    csvDataset
    .map(({xs, ys}) =>
      {
        // Convert xs(features) and ys(labels) from object form (keyed by
        // column name) to array form.
        return {xs:Object.values(xs), ys:Object.values(ys)};
      })
    .batch(10);
}

run();


function normal() {
    var x = 0,
        y = 0,
        rds, c;
    do {
        x = Math.random() * 2 - 1;
        y = Math.random() * 2 - 1;
        rds = x * x + y * y;
    } while (rds == 0 || rds > 1);
    c = Math.sqrt(-2 * Math.log(rds) / rds); // Box-Muller transform
    return x * c; // throw away extra sample y * c
}

var N = 2000,
  a = -1,
  b = 1.2;

var step = (b - a) / (N - 1);
var t = new Array(N), x = new Array(N), y = new Array(N);

for(var i = 0; i < N; i++){
  t[i] = a + step * i;
  x[i] = (Math.pow(t[i], 1))*1 + (.3 * normal() );
  y[i] = (Math.pow(t[i], 1))*1 + (.3 * normal() );

  x[i] = [x[i],y[i]];
}


let data0 =   tf.randomNormal([1,500],0,1).transpose();
let data1 = tf.randomNormal([1,500],0,1).transpose();



let data = data0.concat(data1,axis=1).arraySync();

PCA({x: x ,y:[]})