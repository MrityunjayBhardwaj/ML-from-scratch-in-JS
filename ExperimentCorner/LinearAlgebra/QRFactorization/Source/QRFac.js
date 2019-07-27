
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


function modifiedGramSchmidt(A){
    
    const Q = A.slice([0,0],[-1,1]);
    const r = tf.zeros([A.shape[1],A.shape[1]]).arraySync();

    const V = tf.tensor(tf.zeors([A.shape[0],1]));

    for (let i=0;i<A.shape[1];i++){
        const v_i = A.slice([0,i], [-1,1]);

        r[i][i] = pNorm(v_i, p=2);
        const q_i = v_i.div(r[i][i]);

        Q.concat(q_i, axis=1);

        for(let j = i+1;j<A.shape[1];j++){
            let v_j = A.slice([0,j], [-1,1]);
            r[i][j] = q_i.matMul(v_j);

            v_j = v_j.sub(q_i.mul(r[i][j]))

            V.concat(v_j, axis=1);
        }
    }

    return [Q.slice([0,1],[-1,-1]),tf.tensor(r)]
}


function mgs(A){
    const V = A;
    const r = tf.zeros([A.shape[1],A.shape[1]]).arraySync();

    let Q = 0;

    const nV = 0;
    for(let i=0;i<A.shape[1];i++){
        const v_i = V.slice([0,i],[-1,1])
        r[i][i] = pNorm(v_i,p=2);

        // q_i = Q.slice([0, 0],[-1, 1]);
        const q_i = v_i.div(r[i][i]);

        if (!Q)Q = q_i
        else{
            Q = Q.concat(q_i, axis=1);
        }

        for(let j =(i+1);j< A.shape[1]){

            const v_j = V.slice([0, j],[0, 1])

            r[i][j] = q_i.matMul(v_i);
            v_j = v_j.sub(q_i.mul(r[i][j]));

            // if (!nV){  }
            V = V.concat(v_j, axis=1);
        }

        return [Q, tf.tensor(r)];

    }

}



function householderQR(A){
    const {m,n} = A.shape();

    const Q = tf.eye(m);

    for(let k =0;k<n;k++){
        // find the reflector for curr col
        const z = A.slice([k,k],[-1,1]);

        let v = sign(z.slice([0,0],[1,1])).mul(pNorm(z, p=2)).sub( (z.slice([0,0],[1,1])) )
    }
}