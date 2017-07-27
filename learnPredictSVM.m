function [C, prob_estimates] = learnPredictSVM(tr_features, ts_features, tr_labels, ts_labels)

if isempty(tr_features)
  C = nan;
  return
end

% check dimensionality
if ndims(tr_features) == 3,
  tr_features = reshape( ...
    tr_features, ...
    [size(tr_features, 1)*size(tr_features, 2), size(tr_features, 3)] ...
  );
  ts_features = reshape( ...
    ts_features, ...
    [size(ts_features, 1)*size(ts_features, 2), size(ts_features, 3)] ...
  );
end

if size(tr_features, 1) ~= size(tr_features, 2)
  if size(tr_features, 1) ~= length(tr_labels), 
    tr_features = tr_features'; 
    ts_features = ts_features';
  end
elseif size(ts_features, 1) ~= size(ts_features, 2)
  if size(ts_features, 1) ~= length(ts_labels), 
    tr_features = tr_features'; 
    ts_features = ts_features';
  end
else
  warning('please cleck the input, dimensionality may be inconsistent')
end
  
if nargin < 4
  ts_labels = zeros(size(ts_features, 1), 1);
end

tr_labels = tr_labels(:);
ts_labels = ts_labels(:);

% train SVM on source training data
model = train(tr_labels, sparse(double(tr_features)), '-s 5 -c 1 -B 1 -q');

% predict SVM on target test data
[C, ~, prob_estimates] = predict(ts_labels, sparse(double(ts_features)), model, '-q');

