function annotations = genAnnotations(opt)

numClass = opt.nclasses;

sourceName = opt.sourcedir; 
targetName = opt.targetdir;

if opt.nclasstrain > 0
  % sampling protocol
  numTrials = opt.ntrials;
  numTrainSource = opt.nclasstrain;
  if strcmp(sourceName, 'dslr') || strcmp(sourceName, 'webcam')
    numTrainSource = 8;
  end
else
  % full protocol
  numTrials = 1;
  numTrainSource = realmax;
end

if exist(fullfile(opt.annotationdir, [sourceName '-' targetName '_' num2str(numTrials) 'RandomTrials' '_' num2str(numClass) 'Categories.mat']), 'file')
  load(fullfile(opt.annotationdir, [sourceName '-' targetName '_' num2str(numTrials) 'RandomTrials' '_' num2str(numClass) 'Categories']))
  return
end

sourceImagePath = fullfile(opt.datasetdir, opt.sourcedir, opt.imagedir);
targetImagePath = fullfile(opt.datasetdir, opt.targetdir, opt.imagedir);

% source domain setting
imNameSource = cell(1, numClass);
idxTrainSource = cell(1, numTrials);
categories = cell(1, numClass); 
for i = 1:numClass
  className = opt.classes{i};
  categories{i} = className;
  
  imPathClassSource = fullfile(sourceImagePath, className);
  imList = dir(fullfile(imPathClassSource, '*.jpg'));
  
  imName = cell(1, length(imList));
  for j = 1:length(imList)
    imName{j} = imList(j).name;
  end
  imNameSource{i} = imName;
  
  numIm = length(imList);
  numPick = min(numIm, numTrainSource);
  for k = 1:numTrials
    idx = randperm(numIm, numPick);
    idxTrainSource{k}{i} = idx; % random sampling a subset of data for training
  end
end

% target domain setting
imNameTarget = cell(1, numClass);
idxTrainTestTarget = cell(1, numTrials);
for i = 1:numClass
    className = opt.classes{i};

    imPathClassTarget = fullfile(targetImagePath, className);
    imList = dir(fullfile(imPathClassTarget, '*.jpg'));

    imName = cell(1, length(imList));
    for j = 1:length(imList)
      imName{j} = imList(j).name;
    end
    imNameTarget{i} = imName;

    for k = 1:numTrials
      idx = 1:length(imList);
      idxTrainTestTarget{k}{i} = idx;
    end
end

annotations.prm.sourceName = sourceName;
annotations.prm.targetName = targetName;
annotations.prm.categories = categories;
annotations.prm.numTrainSource = numTrainSource;

annotations.imagenames.source = imNameSource;
annotations.imagenames.target = imNameTarget;

annotations.train.source = idxTrainSource;

annotations.train.target = idxTrainTestTarget;
annotations.test.target = idxTrainTestTarget;

% save annotations
save(fullfile(opt.annotationdir, [sourceName '-' targetName '_' num2str(numTrials) 'RandomTrials' '_' num2str(numClass) 'Categories']), 'annotations')

