function X = hl_parse_fea1d(X, opt)

switch opt.featureType
  case 'CONV'
    if ndims(X) == 4
      X = permute(X, [3 1 2 4]);
      [m, n, p, q] = size(X);
      X = reshape(X, [m*n*p, q]);
    end
  otherwise
    % do nothing
end