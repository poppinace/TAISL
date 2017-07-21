% This is a demo code for
% When Unsupervised Domain Adaptation Meets Tensor Rrepresentations
% H. Lu, L. Zhang, Z. Cao, W. Wei, K. Xian, C. Shen, and A. van den Hengel
% IEEE International Conference on Computer Vision (ICCV), 2017
%
% Contact: Hao Lu (poppinace@hust.edu.cn, poppinace@foxmail.com)

clear; close all; clc

warning('off') %#ok<WNOFF>

addpath('liblinear-2.1/matlab');
addpath(genpath('./tensor_toolbox_2.6'));
addpath('FOptM-share-v0.1');

% set seed
rng('default')

% parameter initialization
opt = paramInit;

% generate annotations
annotations = genAnnotations(opt);

% train and test domain adaptation
[meanAcc, stdAcc] = trainTestDA(opt, annotations);

acc{1} = {meanAcc.na, stdAcc.na};
acc{2} = {meanAcc.ntsl, stdAcc.ntsl};
acc{3} = {meanAcc.taisl, stdAcc.taisl};
  
% print results
print_on_screen(acc, annotations)