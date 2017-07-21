# Tensor-Aligned Invariant Subspace Learning
**When Unsupervised Domain Adaptation Meets Tensor Representations**

Proc. IEEE International Conference on Computer Vision (ICCV), 2017

By [Hao Lu](https://sites.google.com/site/poppinace/)<sup>1</sup>, [Lei Zhang](https://sites.google.com/site/leizhanghyperspectral/)<sup>2</sup>, Zhiguo Cao<sup>1</sup>, Wei Wei<sup>2</sup>, Ke Xian<sup>1</sup>, [Chunhua Shen](http://cs.adelaide.edu.au/~chhshen/)<sup>3</sup>, [Anton van den Hengel](https://cs.adelaide.edu.au/~hengel/)<sup>3</sup>

<sup>1</sup>Huazhong University of Science and Technology, China

<sup>2</sup>Northwestern Polytechnical University, China

<sup>3</sup>The University of Adelaide, Australia


### Introduction

This repository contains the implimentation of Naive Tensor Subspace Learning (NTSL) and Tensor-Aligned Invariant Subspace Learning (TAISL) proposed in our ICCV17 paper.

**Prerequisites**
1. Matlab is required. This repository has been tested on Mac OS X Matlab2016a. It should also be compatible with Windows 10.
2. LibLinear toolbox at: https://www.csie.ntu.edu.tw/~cjlin/liblinear/. Please remember to install it following the instruction on the website, especially for Windows and Ubantun users.
3. Tensor Toolbox at: http://www.sandia.gov/~tgkolda/TensorToolbox/index-2.6.html.
4. Matlab code for optimization with orthogonality constraints at: http://optman.blogs.rice.edu.

For your convenience, these toolboxs have already been included in this repository. Please remember to cite corresponding papers/softwares if you use these codes.

**Usage**
1. run demo.m for a demonstration for the domain adaptation task of W->C.

### Citation

If you use our codes in your research, please cite:

	@inproceedings{Hao2017,
		author = {Hao Lu and Lei Zhang and Zhiguo Cao and Wei Wei and Ke Xian and Chunhua Shen and Anton van den Hengel},
		title = {When Unsupervised Domain Adaptation Meets Tensor Representations},
		booktitle = {Proc. IEEE International Conference on Computer Vision (ICCV)},
		year = {2017}
	}
