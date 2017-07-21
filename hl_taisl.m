function [S, T] = hl_taisl(Xs, Xt, Ys, Yt, opt)
%HL_TAISL implements Tensor-Aligned Invariant Subspace Learning proposed in
% When Unsupervised Domain Adaptation Meets Tensor Rrepresentations
% H. Lu, L. Zhang, Z. Cao, W. Wei, K. Xian, C. Shen, and A. van den Hengel
% IEEE International Conference on Computer Vision (ICCV), 2017
%
% Contact: Hao Lu (poppinace@hust.edu.cn, poppinace@foxmail.com)

taisl = opt.taisl;

d = taisl.d;
lambda = taisl.lambda;
maxiter = taisl.maxiter;
maxiter_orth = taisl.maxiter_orth;
mtol = taisl.mtol;
utol = taisl.utol;

proj_flag = taisl.proj_flag;
record = taisl.record;
verbose = taisl.verbose;

Nk = ndims(Xs);

if strcmp(proj_flag, 'spatial')
  nmode = Nk - 2;
elseif strcmp(proj_flag, 'complete')
  nmode = Nk - 1;
end
  
Xs_ = tensor(Xs);
Xt_ = tensor(Xt);

% intialize U
X_ = tensor(cat(Nk, Xs, Xt));
U = tucker_als(X_, d, 'init', 'nvecs', 'printitn', 0);

% initialize M
if Nk == 3
  [hs, ds, ns] = size(Xs);
  M = {eye(hs), eye(ds)};
elseif Nk == 4
  [hs, ws, ds, ns] = size(Xs);
  M = {eye(hs), eye(ws), eye(ds)};
elseif Nk > 4
  error('higher-order is currently not supported')
end

idx = randperm(ns);
ns2 = round(ns/2);
if Nk == 3
  valXs1_ = Xs_(:, :, idx(1:ns2));
  valXs2_ = Xs_(:, :, idx(ns2+1:end));
elseif Nk == 4
  valXs1_ = Xs_(:, :, :, idx(1:ns2));
  valXs2_ = Xs_(:, :, :, idx(ns2+1:end));
end
valYs1 = Ys(idx(1:ns2));
valYs2 = Ys(idx(ns2+1:end));

% compute initial loss and accuracy
if verbose
  loss0 = lossfun(Xs_, Xt_, M, U, lambda);
  acc_val0 = eval_acc(valXs1_, valXs2_, valYs1, valYs2, M, U, proj_flag);
  acc0 = eval_acc(Xs_, Xt_, Ys, Yt, M, U, proj_flag);
  fprintf( ...
    '\niter 0, loss = %.2f, acc_val = %.2f, acc = %.2f\n', ...
    loss0, ...
    acc_val0, ...
    acc0 ...
  )
end

% set options for orthogonality minimizer
opts.mxitr = maxiter_orth;
opts.record = record;

% alternative minimization
loss = zeros(2, maxiter);
acc = zeros(2, maxiter);
acc_val = zeros(2, maxiter);
deltaM = zeros(1, maxiter);
deltaU = zeros(1, maxiter);
for i = 1:maxiter
  Up = U;
  Mp = M;
  
  % update M given U
  G_ = U.core;
  Us = U.U;
  Us{Nk} = Us{Nk}(1:ns, :);
  Y_ = ttm(G_, Us, 1:Nk);
  for k = 1:nmode
    Xk = unfolding_by_k(Xs_, k);
    Yk = unfolding_by_k(Y_, k);
    M_k = folding_exclude_k(M, k);
    QQt = zeros(size(Xk, 1), size(Xk, 1));
    QYt = zeros(size(Xk, 1), size(Yk, 1));
    YYt = zeros(size(Yk, 1), size(Yk, 1));
    XXt = zeros(size(Xk, 1), size(Xk, 1));
    for n = 1:ns
      Qn = Xk(:, :, n) * M_k';
      QQt = QQt + Qn * Qn';
      QYt = QYt + Qn * Yk(:, :, n)';
      YYt = YYt + Yk(:, :, n) * Yk(:, :, n)';
      XXt = XXt + Xk(:, :, n) * Xk(:, :, n)';
    end
    P = orth_min(M{k}', QQt, QYt, YYt, XXt, opts, -lambda);
    M{k} = P';
  end
  
  % evaluate loss and accuracy
  if verbose
    loss(1, i) = lossfun(Xs_, Xt_, M, U, lambda);
    acc_val(1, i) = eval_acc(valXs1_, valXs2_, valYs1, valYs2, M, U, proj_flag);
    acc(1, i) = eval_acc(Xs_, Xt_, Ys, Yt, M, U, proj_flag);
    deltaM(i) = sum(cellfun(@(A, B) norm(A - B, 'fro') / norm(B, 'fro'), M, Mp)) / nmode;
    fprintf( ...
      'M step: iter %d, loss = %.2f, acc_val = %.2f, acc = %.2f, deltaM = %.2e\n', ...
      i, ...
      loss(1, i), ...
      acc_val(1, i), ...
      acc(1, i), ...
      deltaM(i) ...
    )
  end
  
  % update U given M
  Z_ = ttm(Xs_, M, 1:Nk-1);
  X_ = tensor(cat(Nk, double(Z_), Xt));
  U = tucker_als(X_, d, 'init', Up.U, 'printitn', 0);
  
  % evaluate loss and accuracy
  if verbose
    loss(2, i) = lossfun(Xs_, Xt_, M, U, lambda);
    acc_val(2, i) = eval_acc(valXs1_, valXs2_, valYs1, valYs2, M, U, proj_flag);
    acc(2, i) = eval_acc(Xs_, Xt_, Ys, Yt, M, U, proj_flag);
    deltaU(i) = sum(cellfun(@(A, B) norm(A - B, 'fro') / norm(B, 'fro'), U.U(1:Nk-1), Up.U(1:Nk-1))) / 3;
    fprintf( ...
      'U step: iter %d, loss = %.2f, acc_val = %.2f, acc = %.2f, deltaU = %.2e\n', ...
      i, ...
      loss(2, i), ...
      acc_val(2, i), ...
      acc(2, i), ...
      deltaU(i) ...
    )
  end
  
  % check for convergence
  dM = sum(cellfun(@(A, B) norm(A - B, 'fro') / norm(B, 'fro'), M, Mp)) / nmode;
  dU = sum(cellfun(@(A, B) norm(A - B, 'fro') / norm(B, 'fro'), U.U(1:Nk-1), Up.U(1:Nk-1))) / 3;
  if dM < mtol || dU < utol
    break
  end
end

Z_ = ttm(Xs_, M, 1:Nk-1);
S = reverse_proj(Z_, U.U, proj_flag);
T = reverse_proj(Xt_, U.U, proj_flag);

end

function [X, out] = orth_min(X, QQt, QYt, YYt, XXt, opts, lambda)
% min 1/2\sum_n||A_n * X - Y_n||^2_F + lam/2 * \sum_n||B_n * X||^2_F, S.t., X'*X = I_k, where X \in R^{n,k}

if ~isfield(opts, 'record'), opts.record = true; end
if ~isfield(opts, 'mxitr'), opts.mxitr = 200; end
opts.xtol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;

[X, out]= OptStiefelGBB(X, @fun, opts, QQt, QYt, YYt, XXt, lambda);

end

function [F, G] = fun(P, QQt, QYt, YYt, XXt, lambda)
% fun:  1/2 * \sum_n|| A_n * X - Y_n||^2_F + lam/2 * \sum_n||B_n * X||^2_F
% gradient : \sum_n A_n' * (A_n * X - Y_n) + lam * \sum_n B_n' * B_n * X
%

tr = @(A) sum(diag(A));

F = (1 / 2) * tr(P' * QQt * P) - tr(P' * QYt) + ...
  (1 / 2) * tr(YYt) - (lambda / 2) * tr(P' * XXt * P);

G = QQt * P - QYt - lambda * XXt * P;

end

function F = lossfun(Xs_, Xt_, M, U, lambda)
  Nk = ndims(Xs_);
  ns = size(Xs_, Nk);
  G_ = U.core;
  Us = U.U;
  Us{Nk} = Us{Nk}(1:ns, :);
  Ut = U.U;
  Ut{Nk} = Ut{Nk}(ns+1:end, :);
  
  Z_ = ttm(Xs_, M, 1:Nk-1);
  Y_ = ttm(G_, Us, 1:Nk);
  W_ = ttm(G_, Ut, 1:Nk);
  
  fro_s = norm(Z_ - Y_)^2;
  fro_t = norm(Xt_ - W_)^2;
  reg = norm(ttm(Z_, M', 1:Nk-1) - Xs_)^2;
  F = fro_s + fro_t + (lambda / 2) * reg;
end

function acc = eval_acc(Xs_, Xt_, Ys, Yt, M, U, proj_flag)
  Nk = ndims(Xs_);
  Z_ = ttm(Xs_, M, 1:Nk-1);
  S = reverse_proj(Z_, U.U, proj_flag);
  T = reverse_proj(Xt_, U.U, proj_flag);
  C = learnPredictSVM(S, T, Ys, Yt);
  acc = normAcc(Yt, C);
end

function Mk = folding_exclude_k(M_, k)

K = length(M_);
if K == 2
  idxK = 1:K;
  idxK(k) = [];
  Mk = M_{idxK};
else
  idxK = 1:K;
  idxK(k) = [];
  for i = K-1:-1:2;
    if i == K-1
      Mk = kron(M_{idxK(i)}, M_{idxK(i-1)});
    else
      Mk = kron(Mk, M_{idxK(i)});
    end
  end
end

end

function Vk = unfolding_by_k(V_, k)

Nk = ndims(V_);
sz = size(V_);

if Nk == 3
  switch k
    case 1
      Vk = reshape(V_, [sz(1), sz(2), sz(3)]);
    case 2
      V_ = permute(V_, [2 1 3]);
      Vk = reshape(V_, [sz(2), sz(1), sz(3)]);
    otherwise
      error('unsupported unfolding of mode k')
  end
elseif Nk == 4
  switch k
    case 1
      Vk = reshape(V_, [sz(1), sz(2)*sz(3), sz(4)]);
    case 2
      V_ = permute(V_, [2 1 3 4]);
      Vk = reshape(V_, [sz(2), sz(1)*sz(3), sz(4)]);
    case 3
      V_ = permute(V_, [3 1 2 4]);
      Vk = reshape(V_, [sz(3), sz(1)*sz(2), sz(4)]);
    otherwise
      error('unsupported unfolding of mode k')
  end
end
Vk = double(Vk);

end

function G = reverse_proj(X_, U, proj_flag)
  Nk = ndims(X_);
  switch proj_flag
    case 'spatial'
      G_ = ttm(X_, U, 1:Nk-2, 't');
    case 'feature'
      G_ = ttm(X_, U, Nk-1, 't');
    case 'complete'
      G_ = ttm(X_, U, 1:Nk-1, 't');
    otherwise
      error('unsupported projection flag')
  end
  G = double(G_);
  if Nk == 3
    [H, D, N] = size(G);
    G = reshape(G, [H*D, N]);
  elseif Nk == 4
    [H, W, D, N] = size(G);
    G = reshape(G, [H*W*D, N]);
  end
end
