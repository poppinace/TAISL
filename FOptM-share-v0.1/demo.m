function demo

m = 50;
n = 500;
X = rand(m,n);
M0 = randn(m,m);
M0 = orth(M0);
Y = M0 * X;

opts.record = 1; %
opts.mxitr  = 100;
opts.xtol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;

M = randn(m,m);    
M = orth(M);
lam = 0.01;
tic;
%[M, out]= OptStiefelGBB(M, @fun, opts, X, Y, lam);
%[M, out]= OptStiefelGBB(M, @fun2, opts, X, Y);
[M] = OrthOptmization(A, Y, B, lam)
tsolve = toc;
out.fval = -2*out.fval; % convert the function value to the sum of eigenvalues
fprintf('\nOptM: obj: %7.6e, itr: %d, nfe: %d, cpu: %f, norm(XT*X-I): %3.2e \n', ...
    out.fval, out.itr, out.nfe, tsolve, norm(M'*M - eye(m), 'fro') );

end

function [F, G] = fun(M, X, Y, lam)
% fun:  || MX - Y||^2_F - lam * ||MX||^2_F
% gradient : [(MX - Y) - lam * MX] * X'
%
T = M * X;
P = T - Y;
F = trace(P' * P) - lam * trace(T' * T); % objective function
G = (P - lam * T) * X';% gradient
end

function [F, G] = fun2(M, X, Y)
% fun:  || MX - Y||^2_F
% gradient : (MX - Y) * X'
%
T = M * X;
P = T - Y;
F = trace(P' * P); % objective function
G = P * X';% gradient
end