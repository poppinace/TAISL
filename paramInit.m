function opt = paramInit

opt.rootdir = './'; % revise if needed

opt.dataset = 'Office-Caltech10';
opt.modelType = 'VGG-VD-16';
opt.featureType = 'CONV';

% path seting
opt.datasetdir = fullfile(opt.rootdir, 'data', opt.dataset); 

% set annotation path
opt.annotationdir = fullfile(opt.datasetdir, 'annotations');

% set source domain and target domain
opt.sourcedir = 'dslr';
opt.targetdir = 'caltech';

opt.nclasstrain = 20; % number of samples per class sampled
opt.ntrials = 20; % number of trials

opt.imagedir = 'images';

opt.cachedir = 'cache';

switch opt.dataset
  case 'Office-Caltech10'
    opt.classes={...
        'back_pack'
        'bike'
        'calculator'
        'headphones'
        'keyboard'
        'laptop_computer'
        'monitor'
        'mouse'
        'mug'
        'projector'
    };
  otherwise
    error('Unsupported dataset')
end

% parameter setting
opt.nclasses = length(opt.classes);

% ntsl setting
opt.ntsl.proj_flag = 'complete'; % 'spatial', 'feature', 'complete'
opt.ntsl.d = [6, 6, 128, 20];

% tasl_orth setting
opt.taisl.proj_flag = 'complete'; % 'spatial', 'complete'
opt.taisl.d = [6, 6, 128, 20];
opt.taisl.lambda = 1e-5;
opt.taisl.maxiter = 5;
opt.taisl.maxiter_orth = 100;
opt.taisl.mtol = 5e-2;
opt.taisl.utol = 5e-2;
opt.taisl.record = false;
opt.taisl.verbose = false;

end