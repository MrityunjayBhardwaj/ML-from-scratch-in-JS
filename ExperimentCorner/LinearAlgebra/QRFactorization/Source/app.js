let myA = ([[1,3], [1,-1], [-2,1]]);
myA = [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]];

const A = tf.tensor(myA)
// const Q = qrFac(A.transpose())

// const w = classicalGramSchmidt(A.transpose());

// w[0].print();
// w[1].print();
// nd.array(myA).qr()

const b = householder(A);
const c = householder_lots(A);

// const d = implicit_Qx()