function test_eig_rand_demo

%-------------------------------------------------------------
% A demo of solving
%   min f(X), s.t., X'*X = I, where X is an n-by-p matrix
%
%  This demo solves the eigenvalue problem by letting
%  f(X) = -0.5*Tr(X'*A*X);
%
%  The result is compared to the MATLAB function "eigs",
%  which call ARPACK (FORTRAN) to find leading eigenvalues.
%
%  Our solver can be faster when n is large and p is small
%
%  The advantage of our solver is not obvious in this demo 
%  since our solver is a general MATLAB code while ARPACK implemented
%  many tricks for computing the eigenvalues.
%-------------------------------------------------------------

clc

seed = 2010;
fprintf('seed: %d\n', seed);
if exist('RandStream','file')
   RandStream.setDefaultStream(RandStream('mt19937ar','seed',seed));
else
   rand('state',seed); randn('state',seed^2);
end

% nlist = [500, 1000, 2000, 3000, 4000, 5000];
nlist = [2000];
nlen = length(nlist);

perf = zeros(10,nlen);

for dn = 1:nlen
    n = nlist(dn);
    fprintf('matrix size: %d\n', nlist(dn));
    
    A = randn(n); A = A'*A;
    k = 6;
    opteig.issym = 1;
    nAx = 0;
    
    % --- MATLAB eigs ---
    tic; [V, D] = eigs(@funAX, n, k, 'la',opteig); teig = toc; D = diag(D); feig = sum(D(1:k));
    
    fprintf('\neigs: obj val %7.6e, cpu %f, #func eval %d\n', feig, teig, nAx);
    feasi = norm(V'*V - eye(k), 'fro');
    
    % --- our solver ---
    % X0 = eye(n,k);
    X0 = randn(n,k);    X0 = orth(X0);
    
    opts.record = 0;
    opts.mxitr  = 1000;
    opts.xtol = 1e-5;
    opts.gtol = 1e-5;
    opts.ftol = 1e-8;
    out.tau = 1e-3;
    %opts.nt = 1;
    
    %profile on;
    tic; [X, out]= OptStiefelGBB(X0, @funeigsym, opts, A); tsolve = toc;
    %profile viewer;
    
    % profile viewer;
    out.fval = -2*out.fval;
    err = (feig-out.fval)/(abs(feig)+1);
    fprintf('ours: obj val %7.6e, cpu %f, #func eval %d, itr %d, |XT*X-I| %3.2e\n', ...
             out.fval, tsolve, out.nfe, out.itr, norm(X'*X - eye(k), 'fro'));
    fprintf('relative difference between two obj vals: %3.2e\n',...
         err);
    out.feasi = norm(X'*X - eye(k), 'fro');
    
    
    perf(:,dn) = [feig;   feasi; teig; nAx; out.fval;  out.feasi; out.nrmG; out.nfe; tsolve; err];
    
end
% save('results/eig_rand_perf', 'perf', 'nlist');


    function AX = funAX(X)
        nAx = nAx + 1;
        AX = A*X;
        %fprintf('iter: %d, size: (%d, %d)\n', nAx, size(X));
    end

    function [F, G] = funeigsym(X,  A)
        
        G = -(A*X);
        %F = 0.5*sum(sum( G.*X ));
        F = 0.5*sum(dot(G,X,1));
        % F = sum(sum( G.*X ));
        % G = 2*G;
        
    end


end
