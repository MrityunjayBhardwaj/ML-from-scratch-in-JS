const myA = ([[1,3], [1,-1], [-2,1]]);
const A = tf.tensor(myA)
const Q = qrFac(A.transpose())

const w = classicalGramSchmidt(A.transpose());

w[0].print();
w[1].print();
// nd.array(myA).qr()