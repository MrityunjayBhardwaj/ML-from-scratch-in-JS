// def LU(A):
//     U = np.copy(A)
//     m, n = A.shape
//     L = np.eye(n)
//     for k in range(n-1):
//         for j in range(k+1,n):
//             L[j,k] = U[j,k]/U[k,k]
//             U[j,k:n] -= L[j,k] * U[k,k:n]
//     return L, U


function LU(A){
    const {0:m, 1:n} = A.shape;

    const U = A;
    L = tf.eye(n);

    for(let k=0;k<n-1;k++){
        for(let j=k+1;j<n;j++){

            const L_jk = L.slice([j,k],[1,1]);
            const U_jk = U.slice([j,k],[1,1]);
            const U_kk = U.slice([k,k],[1,1]);

            const newL_jk = U_jk.div( U_kk );
            L = replace2Tensor(L,newL_jk,[j,k]);

            const U_jk2n = U.slice([j,k],[1,-1]);
            const U_kk2n = U.slice([k,k],[1,-1]);

            const newU_jk2n = U_jk2n.sub(L_jk.matMul(U_kk2n));

            U = replace2Tensor(U, newU_jk2n, [j,k]);
        }
    }

    return { L:L, U:U };
}


// gaussian elimination with partial pivoting

// % Gaussian elimination with partial pivoting.
// for j=1:n-1  % Loop over columns.
//   [pivot,k] = max(abs(A(j:n,j))); % Find the pivot element
//                                   % in column j.
//                                   % pivot is the largest
//                                   % absolute value of an
//                                   % entry; k+j-1 is its index.
//   if pivot==0,                    % If all entries in the
//                                   % column are 0, return with
//     disp(' Matrix is singular.')   % an error message.
//     break;
//   end;
//   temp = A(j,:);            % Otherwise,
//   A(j,:) = A(k+j-1,:);      % Interchange rows j and k+j-1.
//   A(k+j-1,:) = temp;
//   tempb = b(j);
//   b(j) = b(k+j-1);
// b(k+j-1) = tempb;
//   for i=j+1:n             % Loop over rows below j.
//     mult = A(i,j)/A(j,j); % Subtract this multiple
//                           % of row j from row
//                           % i to make A(i,j)=0.
//     A(i,j:n) = A(i,j:n) - multimageA(j,j:n);
//     b(i) = b(i) - multimageb(j);
//   end;
// end;
