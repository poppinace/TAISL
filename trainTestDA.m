function [meanAcc, stdAcc] = trainTestDA(opt, annotations)

dataset = opt.dataset;
featureType = opt.featureType;
modelType = opt.modelType;

sourceName = annotations.prm.sourceName; 
targetName = annotations.prm.targetName;

nclasses = opt.nclasses;
ntrials = opt.ntrials;
if opt.nclasstrain < 0
  ntrials = 1;
end

cachedir = opt.cachedir;

EN_NA    = 1;
EN_NTSL  = 1;
EN_TAISL = 1;

% load source features
source = load( ...
    fullfile( ...
      cachedir, ...
      [dataset '_' sourceName '_' modelType '_features_' featureType] ...
    ) ...
  );

% extract target features
target = load( ...
  fullfile( ...
    cachedir, ...
    [dataset '_' targetName '_' modelType '_features_' featureType] ...
  ) ...
);

% parse features
for i = 1:nclasses
  source.features{i} = cat(4, source.features{i}{:});
  target.features{i} = cat(4, target.features{i}{:});
end

% train & test
idxSourceTrials = annotations.train.source;
idxTargetTrials = annotations.test.target;

acc_na = zeros(1, ntrials);
acc_ntsl = zeros(1, ntrials);
acc_taisl = zeros(1, ntrials);

tic;
for i = 1:ntrials
%   warning('off') %#ok<WNOFF>
  
  fprintf('.')
  idxSource = idxSourceTrials{i};
  idxTarget = idxTargetTrials{i};
  % prase training and test data from both domain
  feats_source = [];
  feats_target = [];
  labels_source = [];
  labels_target = [];
  for j = 1:nclasses
    % index
    idxClassSource = idxSource{j};
    idxClassTarget = idxTarget{j};
    % features
    feats_source = cat( ...
      4, ...
      feats_source, ...
      source.features{j}(:, :, :, idxClassSource) ...
    );
    feats_target = cat( ...
      4, ...
      feats_target, ...
      target.features{j}(:, :, :, idxClassTarget) ...
    );
    % labels
    labels_source = cat( ...
      1, ...
      labels_source, ...
      source.labels{j}(idxClassSource) ...
    );
    labels_target = cat( ...
      1, ...
      labels_target, ...
      target.labels{j}(idxClassTarget) ...
    );
  end
  
% ----------------------------------------------
% baseline no alignment
% ----------------------------------------------
if EN_NA
  [Xs, Xt] = hl_na(feats_source, feats_target, opt);
  Ys = labels_source;
  Yt = labels_target;
  C = learnPredictSVM(Xs, Xt, Ys, Yt);
  acc_na(i) = normAcc(Yt, C);
end

% ----------------------------------------------
% tensor subspace learning
% ----------------------------------------------
if EN_NTSL
  [Xs, Xt] = hl_ntsl(feats_source, feats_target, opt);
  Ys = labels_source;
  Yt = labels_target;
  C = learnPredictSVM(Xs, Xt, Ys, Yt);
  acc_ntsl(i) = normAcc(Yt, C);
end
  
% ----------------------------------------------
% tensor alignment and subspace learning
% ----------------------------------------------
if EN_TAISL
  [Xs, Xt] = hl_taisl(feats_source, feats_target, labels_source, labels_target, opt);
  Ys = labels_source;
  Yt = labels_target;
  C = learnPredictSVM(Xs, Xt, Ys, Yt);
  acc_taisl(i) = normAcc(Yt, C);
end

end

time = toc;
fprintf('\nAverage time for each trial is %4.2f\n', time / ntrials)

meanAcc.na = mean(acc_na); stdAcc.na = std(acc_na);
meanAcc.ntsl = mean(acc_ntsl); stdAcc.ntsl = std(acc_ntsl);
meanAcc.taisl = mean(acc_taisl); stdAcc.taisl = std(acc_taisl);

