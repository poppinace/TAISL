function [S, T] = hl_ntsl(Xs, Xt, opt)
%HL_NTSL implements Naive Tensor Subspace Learning proposed in
% When Unsupervised Domain Adaptation Meets Tensor Rrepresentations
% H. Lu, L. Zhang, Z. Cao, W. Wei, K. Xian, C. Shen, and A. van den Hengel
% IEEE International Conference on Computer Vision (ICCV), 2017
%
% Contact: Hao Lu (poppinace@hust.edu.cn, poppinace@foxmail.com)

d = opt.ntsl.d;
proj_flag = opt.ntsl.proj_flag;

% centering has a negative effect on the performance
% X = cat(4, Xs, Xt);
% meanX = mean(X, 4);
% Xs = bsxfun(@minus, Xs, meanX);
% Xt = bsxfun(@minus, Xt, meanX);

Xs_ = tensor(Xs);
Xt_ = tensor(Xt);

% learning U
X_ = tensor(cat(ndims(Xs), Xs, Xt));

U = tucker_als(X_, d, 'init', 'nvecs', 'printitn', 0);

% projection
S = reverse_proj(Xs_, U.U, proj_flag);
T = reverse_proj(Xt_, U.U, proj_flag);

end

function G = reverse_proj(X_, U, proj_flag)
  n = ndims(X_);
  switch proj_flag
    case 'spatial'
      G_ = ttm(X_, U, 1:n-2, 't');
    case 'feature'
      G_ = ttm(X_, U, n-1, 't');
    case 'complete'
      G_ = ttm(X_, U, 1:n-1, 't');
    otherwise
      error('unsupported projection flag')
  end
  G = double(G_);
  if n == 4
    [H, W, D, N] = size(G);
    G = reshape(G, [H*W*D, N]);
  elseif n == 3
    [H, D, N] = size(G);
    G = reshape(G, [H*D, N]);
  end
end