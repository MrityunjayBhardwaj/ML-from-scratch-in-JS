
function qrFac(A){

    let Q = A.slice([0,0],[-1,1]);
    // let r = tf.zeros(A.shape[1],A.shape[1]).arraySync();
    pNorm(Q,2).print();

    Q = Q.div(pNorm(Q,2));

    console.log(A.shape)

    // let rDiag
    for(let i =1; i< A.shape[1]-1;i++){

        // extract the column vector
        const cColVec = A.slice([0,i], [-1,1]);

        // project current column vector onto Q
        const projColVec = ndProject(Q,cColVec);

        const orthoVec = cColVec.sub(projColVec);

        Q = Q.concat(tf.div(orthoVec, pNorm(orthoVec,2)), axis=1);

    }

    const R = Q.transpose().matMul(A);

    A.print();

    // reconstruction
    // Q.matMul(R).print();
    R.print();

    console.log("Q: ");
    Q.print();

    return [Q,R];

}


// classical QR decomposition
function classicalGramSchmidt(A){
    let Q = A.slice([0,0],[-1,1]);
    let r = tf.zeros([A.shape[1],A.shape[1]]).arraySync();

    for(let j=0;j<A.shape[1];j++){

        // extract the column vector
        let v_j = A.slice([0,j], [-1,1]);
        // let q_j = Q.slice([0,j], [-1,1]);
        let a_j = A.slice([0,j], [-1,1]);

        for(let i =0;i<(j-1); i++){
            const q_i = Q.slice([0,i], [-1,1]);
            r[i][j] = tf.matMul(q_i.transpose(), a_j).flatten().arraySync()[0];

            v_j = tf.sub(v_j , tf.mul(r[i][j], q_i));
        }
        r[j][j] = pNorm(v_j, p=2).flatten().arraySync()[0];
        q_j = v_j.div(r[j][j]);

        if (!j){Q = q_j; continue;}
        Q = Q.concat(q_j, axis=1);
    }


    return [Q, tf.tensor(r)];
}

// function qrFac(A){

//     const arrayA = A.arraySync();

//     let Q; 
//     for(let j=0;j<A.shape[1];j++){

//             const v_j = arrayA[j];
//             for(let i=0;i<(j-1);i++){
//                 r[i][j] = q[i]*a[j];

//                 v[j] = v[j] - r[i][j]*q[i];
//             }

//             const r_jj = norm(v_j) ;
//             Q[j] = v_J/r_jj; // noramlized v_J
//     }

// }