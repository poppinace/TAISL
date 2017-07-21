function [X, out] = orth_min_sum(A, Y, B, opts, lambda)
% min 1/2\sum_n||A_n * X - Y_n||^2_F + lam/2 * \sum_n||B_n * X||^2_F, S.t., X'*X = I_k, where X \in R^{n,k}

if ~isfield(opts, 'record'), opts.record = 1; end
if ~isfield(opts, 'mxitr'), opts.mxitr = 200; end
opts.xtol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;

[m,n] = size(A(:,:,1));

X = randn(n,n);    
X = orth(X);
[X, out]= OptStiefelGBB(X, @fun, opts, A, Y, B, lambda);

end

function [F, G] = fun(X, A, Y, B, lam)
% fun:  1/2 * \sum_n|| A_n * X - Y_n||^2_F + lam/2 * \sum_n||B_n * X||^2_F
% gradient : \sum_n A_n' * (A_n * X - Y_n) + lam * \sum_n B_n' * B_n * X
%
[m,n,p] = size(A);
F = 0;
G = zeros(n,n);
for i = 1 : p
  T = A(:,:,i) * X - Y(:,:,i);
  P = B(:,:,i) * X;
  F = F + (1 / 2) * trace(T' * T) + (lam / 2) * trace(P' * P); % objective function
  G = G + A(:,:,i)' * T + lam * B(:,:,i)' * P;% gradient
end

end