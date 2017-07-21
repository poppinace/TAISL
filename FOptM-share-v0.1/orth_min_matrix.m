function [X, out] = orth_min(A, Y, B, opts, lambda)
% min 1/2||AX - Y||^2_F + lam/2 * ||BX||^2_F, S.t., X'*X = I_k, where X \in R^{n,k}

if ~isfield(opts, 'record'), opts.record = 1; end
if ~isfield(opts, 'mxitr'), opts.mxitr = 200; end
opts.xtol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;

[m,n] = size(A);

X = randn(n,n);    
X = orth(X);
[X, out]= OptStiefelGBB(X, @fun, opts, A, Y, B, lambda);

end

function [F, G] = fun(X, A, Y, B, lam)
% fun:  1/2 * || AX - Y||^2_F + lam/2 * ||BX||^2_F
% gradient : A' * (AX - Y) + lam * B' * BX
%
T = A * X - Y;
P = B * X;
F = trace(T' * T) + (lam / 2) * trace(P' * P); % objective function
G = A' * T + lam * B' * P;% gradient
end