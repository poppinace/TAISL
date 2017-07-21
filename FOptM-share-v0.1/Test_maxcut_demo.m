function Test_maxcut_demo

%-------------------------------------------------------------
% maxcut SDP: 
% X is n by n matrix
% max Tr(C*X), s.t., X_ii = 1, X psd
%
% low rank model:
% % X = V'*V, V = [V_1, ..., V_n], V is a p by n matrix
% max Tr(C*V'*V), s.t., ||V_i|| = 1,
%
%-------------------------------------------------------------

% clc
clear all

seed = 2010;
fprintf('seed: %d\n', seed);
if exist('RandStream','file')
   RandStream.setGlobalStream(RandStream('mt19937ar','seed',seed));
else
   rand('state',seed); randn('state',seed^2);
end

% change the path here
src = [fileparts(mfilename('fullpath')) '/data/Maxcut/'];

% Probname = {'torusg3-8', 'torusg3-15',  'toruspm3-8-50',  'toruspm3-15-50' };
Probname = { 'torusg3-82', 'torusg3-152',  'toruspm3-8-50',  'toruspm3-15-50', 'G22',  'G23'};
nprob = length(Probname);

% Problist = [1:nprob];
% Problist = [21:40];
% Problist = [31];
Problist = [1];
nlen = length(Problist);

perf = zeros(nprob, 14);

for dprob = Problist;
    %clear n m C
    name = Probname{dprob};
    file = strcat(src,name,'.mat');
    load(file,'n','m','C');

    % modify the estimation of rank here
    p = max(min(round(sqrt(2*n)/2), 20),1);
    
    % initial point should be normalized
    x0 = randn(p,n); nrmx0 = dot(x0,x0,1);   
    x0 = bsxfun(@rdivide, x0, sqrt(nrmx0)); 

    % profile on
    opts.record = 0;
    opts.mxitr  = 600;
    opts.gtol = 1e-5;
    opts.xtol = 1e-5;
    opts.ftol = 1e-8;
    opts.tau = 1e-3;
    tic; [x, g, out]= OptManiMulitBallGBB(x0, @maxcut_quad, opts, C); tsolve = toc;
    %XSDP = x'*x; % 
    objf2 =  -full(out.fval);
    fprintf('name %10s, n %d, p %d, f %6.4e, cpu %4.2f, itr %d, #func eval %d, feasi %3.2e, ||Hx|| %3.2e\n',name, n, p, objf2,  tsolve, out.itr, out.nfe, out.feasi, out.nrmG);
    perf(dprob, 7:14) = [n, p, objf2, tsolve, out.itr, out.nfe, out.feasi, out.nrmG];
    
end

% save('res_maxcut_GLarge_Graph_quad_it600', 'perf');
% save('res_maxcut_torus_Graph_quad_it600', 'perf');



