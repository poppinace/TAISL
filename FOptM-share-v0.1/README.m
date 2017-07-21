% A feasible method for optimization with orthogonality constraints
%
% -------------------------------------------------------------------------
% 1. Problems and solvers
%
% The package contains codes for the following two problems:
% (1) min F(X), s.t., ||X_i||_2 = 1
%
%     Solver: OptManiMulitBallGBB.m
%
%     Solver demo: Test_maxcut_demo.m, solving the max-cut problem
%
% (2) min F(X), S.t., X'*X = I_k, where X \in R^{n,k}
%
%     Solver: OptStiefelGBB.m
%
%     Solver demo: test_eig_rand_demo.m, computing leading eigenvalues
%
% -------------------------------------------------------------------------
% 2. Reference
%
% Zaiwen Wen and Wotao Yin. A Feasible method for Optimization with 
% Orthogonality Constraints, Optimization Online, 11/2010. Also as Rice
% CAAM Tech Report TR10-26.
%
% http://optman.blogs.rice.edu
%
% -------------------------------------------------------------------------
% 3. The Authors
%
% We hope that the package is useful for your application.  If you have
% any bug reports or comments, please feel free to email one of the
% toolbox authors:
%
%   Zaiwen Wen, zw2109@sjtu.edu.cn
%   Wotoa Yin,  wotao.yin@rice.edu
%
% Enjoy!
% Zaiwen and Wotao
%
% -------------------------------------------------------------------------
%  Copyright (C) 2010, Zaiwen Wen and Wotao Yin
% 
%  This program is free software: you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation, either version 3 of the License, or
%  (at your option) any later version.
% 
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
% 
%  You should have received a copy of the GNU General Public License
%  along with this program.  If not, see <http://www.gnu.org/licenses/>
