function [x, g, out]= OptManiMulitBallGBB(x, fun, opts, varargin)
%-------------------------------------------------------------------------
% Line search algorithm for optimization on manifold:
%
%   min f(X), s.t., ||X_i||_2 = 1, where X \in R^{n,p}
%       g(X) = grad f(X)
%   X = [X_1, X_2, ..., X_p]
%
%
%   each column of X lies on a unit sphere
% Input:
%           X --- ||X_i||_2 = 1, each column of X lies on a unit sphere
%         fun --- objective function and its gradient:
%                 [F, G] = fun(X,  data1, data2)
%                 F, G are the objective function value and gradient, repectively
%                 data1, data2 are addtional data, and can be more
%                 Calling syntax:
%                   [X, out]= OptManiMulitBallGBB(X0, @fun, opts, data1, data2);
%
%        opts --- option structure with fields:
%                 record = 0, no print out
%                 mxitr       max number of iterations
%                 xtol        stop control for ||X_k - X_{k-1}||
%                 gtol        stop control for the projected gradient
%                 ftol        stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|)
%                             usually, max{xtol, gtol} > ftol
%   
% Output:
%           x --- solution
%           g --- gradient of x
%         Out --- output information
%
% -------------------------------------
% For example, consider the maxcut SDP: 
% X is n by n matrix
% max Tr(C*X), s.t., X_ii = 1, X psd
%
% low rank model is:
% X = V'*V, V = [V_1, ..., V_n], V is a p by n matrix
% max Tr(C*V'*V), s.t., ||V_i|| = 1,
%
% function [f, g] = maxcut_quad(V, C)
% g = 2*(V*C);
% f = sum(dot(g,V))/2;
% end
%
% [x, g, out]= OptManiMulitBallGBB(x0, @maxcut_quad, opts, C); 
%
% -------------------------------------
%
% Reference: 
%  Z. Wen and W. Yin
%  A feasible method for optimization with orthogonality constraints
%
% Author: Zaiwen Wen, Wotao Yin
%   Version 1.0 .... 2010/10
%-------------------------------------------------------------------------

%% Size information
% termination rule
if isfield(opts, 'xtol')
    if opts.xtol < 0 || opts.xtol > 1
        opts.xtol = 1e-6;
    end
else
    opts.xtol = 1e-6;
end

if isfield(opts, 'ftol')
    if opts.ftol < 0 || opts.ftol > 1
        opts.ftol = 1e-12;
    end
else
    opts.ftol = 1e-12;
end


if isfield(opts, 'gtol')
    if opts.gtol < 0 || opts.gtol > 1
        opts.gtol = 1e-6;
    end
else
    opts.gtol = 1e-6;
end

% parameters for control the linear approximation in line search
if isfield(opts, 'rho')
   if opts.rho < 0 || opts.rho > 1
        opts.rho = 1e-4;
   end
else
    opts.rho = 1e-4;
end

% factor for decreasing the step size in the backtracking line search
if isfield(opts, 'eta')
   if opts.eta < 0 || opts.eta > 1
        opts.eta = 0.1;
   end
else
    opts.eta = 0.2;
end

% parameters for updating C by HongChao, Zhang
if isfield(opts, 'gamma')
   if opts.gamma < 0 || opts.gamma > 1
        opts.gamma = 0.85;
   end
else
    opts.gamma = 0.85;
end



if isfield(opts, 'tau')
   if opts.tau < 0 || opts.tau > 1e3
        opts.tau = 1e-3;
   end
else
    opts.tau = 1e-3;
end


% parameters for the  nonmontone line search by Raydan
if ~isfield(opts, 'M')
    opts.M = 10;
end

if ~isfield(opts, 'STPEPS')
    opts.STPEPS = 1e-10;
end


if isfield(opts, 'nt')
    if opts.nt < 0 || opts.nt > 100
        opts.nt = 5;
    end
else
    opts.nt = 5;
end

if isfield(opts, 'mxitr')
    if opts.mxitr < 0 || opts.mxitr > 2^20
        opts.mxitr = 1000;
    end
else
    opts.mxitr = 1000;
end

if ~isfield(opts, 'record')
    opts.record = 0;
end

%-------------------------------------------------------------------------------
% copy parameters
xtol = opts.xtol;
ftol = opts.ftol;
gtol = opts.gtol;
rho  = opts.rho;
M     = opts.M;
STPEPS = opts.STPEPS;
eta   = opts.eta;
gamma = opts.gamma;

record = opts.record;

nt = opts.nt;
crit = ones(nt, 3);

% normalize x so that ||x||_2 = 1
[n,p] = size(x); nrmx = dot(x,x,1);
if norm(nrmx - 1,'fro')>1e-8; x = bsxfun(@rdivide, x, sqrt(nrmx)); end;

%% Initial function value and gradient
% prepare for iterations
% tmp = cputime; 
[f,g] = feval(fun, x, varargin{:});    out.nfe = 1; 
xtg = dot(x,g,1);   gg = dot(g,g,1);   
xx = dot(x,x,1);    xxgg = xx.*gg;
dtX = bsxfun(@times, xtg, x) - g;    nrmG = norm(dtX, 'fro');

Q = 1; Cval = f; tau = opts.tau;
%% Print iteration header if debug == 1
if (record >= 1)
    fprintf('----------- Gradient Method with Line search ----------- \n');
    fprintf('%4s \t %10s \t %10s \t  %10s \t %5s \t %9s \t %7s \n', 'Iter', 'tau', 'f(X)', 'nrmG', 'Exit', 'funcCount', 'ls-Iter');
    fprintf('%4d \t %3.2e \t %3.2e \t %5d \t %5d	\t %6d	\n', 0, 0, f, 0, 0, 0);
end

if record == 10; out.fvec = f; end

%% main iteration
for itr = 1 : opts.mxitr
    xp = x;     fp = f;     gp = g;   dtXP =  dtX;

    nls = 1; deriv = rho*nrmG^2;
    while 1
        % calculate g, f,
        tau2 = tau/2;     beta = (1+(tau2)^2*(-xtg.^2+xxgg));
        a1 = ((1+tau2*xtg).^2 -(tau2)^2*xxgg)./beta;
        a2 = -tau*xx./beta;
        x = bsxfun(@times, a1, xp) + bsxfun(@times, a2, gp);
        
        %if norm(dot(x,x,1) - 1) > 1e-6
        %    error('norm(x)~=1');
        %end
        
        %f = feval(fun, x, varargin{:});   out.nfe = out.nfe + 1;
        [f,g] = feval(fun, x, varargin{:});   out.nfe = out.nfe + 1;
        
        if f <= Cval - tau*deriv || nls >= 5
            break
        end
        tau = eta*tau;
        nls = nls+1;
    end  
    
    if record == 10; out.fvec = [out.fvec; f]; end
    % evaluate the gradient at the new x
    %[f,g] = feval(fun, x, varargin{:});  %out.nfe = out.nfe + 1;
    
    xtg = dot(x,g,1);   gg = dot(g,g,1);   
    xx = dot(x,x,1);    xxgg = xx.*gg;
    dtX = bsxfun(@times, xtg, x) - g;    nrmG = norm(dtX, 'fro');
    s = x - xp; XDiff = norm(s,'fro')/sqrt(n);
    FDiff = abs(fp-f)/(abs(fp)+1);

    if (record >= 1)
        fprintf('%4d \t %3.2e \t %7.6e \t %3.2e \t %3.2e \t %3.2e \t %2d\n', ...
            itr, tau, f, nrmG, XDiff, FDiff, nls);
    end
    
    crit(itr,:) = [nrmG, XDiff, FDiff];
    mcrit = mean(crit(itr-min(nt,itr)+1:itr, :),1);
    
    %if ((XDiff < xtol) || (nrmG < gtol) ) %|| abs((fp-f)/max(abs([fp,f,1]))) < 1e-20;
    if ( XDiff < xtol && FDiff < ftol ) || nrmG < gtol || all(mcrit(2:3) < 10*[xtol, ftol])  
        out.msg = 'converge';
        break;
    end
    
    %y = g - gp;      
    y = dtX - dtXP;
    sy = sum(sum(s.*y));    tau = opts.tau;
    sy = abs(sy);
    if sy > 0; 
        %tau = sum(sum(s.*s))/sy;
        %tau = sy/sum(sum(y.*y));
        %tau = sum(sum(s.*s))/sy + sy/sum(sum(y.*y));
        if mod(itr,2)==0; tau = sum(sum(s.*s))/sy;
        else tau = sy/sum(sum(y.*y)); end
        
        % safeguarding on tau
        tau = max(min(tau, 1e20), 1e-20);
    end
    Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + f)/Q;
end

if itr >= opts.mxitr
    out.msg = 'exceed max iteration';
end

out.feasi = norm(dot(x,x,1)-1);
if out.feasi > 1e-14
    nrmx = dot(x,x,1);      x = bsxfun(@rdivide, x, sqrt(nrmx)); 
    [f,g] = feval(fun, x, varargin{:});   out.nfe = out.nfe + 1;
    out.feasi = norm(dot(x,x,1)-1);
end

out.nrmG = nrmG;
out.fval = f;
out.itr = itr;



