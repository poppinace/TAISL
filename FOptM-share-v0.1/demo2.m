function demo2

m = 500;
n = 50;
A = rand(m,n);
X0 = randn(n,n);
X0 = orth(X0);
Y = A * X0;

lam = -0.001;
tic;
[X, out] = OrthOptmization(A, Y, A, lam);
tsolve = toc;
out.fval = -2*out.fval; % convert the function value to the sum of eigenvalues
fprintf('\nOptM: obj: %7.6e, itr: %d, nfe: %d, cpu: %f, norm(XT*X-I): %3.2e \n', ...
    out.fval, out.itr, out.nfe, tsolve, norm(X'*X - eye(n), 'fro') );

end