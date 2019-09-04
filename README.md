# [Understanding and Accelerating Particle-Based Variational Inference](http://proceedings.mlr.press/v97/liu19i.html)
[Chang Liu][changliu] \<<chang-li14@mails.tsinghua.edu.cn> (deprecated); <liuchangsmail@gmail.com>\>,
[Jingwei Zhuo][jingweizhuo], Pengyu Cheng, Ruiyi Zhang, [Jun Zhu][junzhu], and [Lawrence Carin][lcarin]. ICML 2019.

\[[Paper & Appendix](http://ml.cs.tsinghua.edu.cn/~changliu/awgf/AWGF.pdf)\]
\[[Slides](http://ml.cs.tsinghua.edu.cn/~changliu/awgf/AWGF_beamer.pdf)\]
\[[Poster](http://ml.cs.tsinghua.edu.cn/~changliu/awgf/AWGF_poster.pdf)\]

## Introduction

The project aims at understanding the mechanism of particle-based variational inference methods
(ParVIs; e.g., Stein Variational Gradient Descent (SVGD; [Liu & Wang, 2016][svgd-paper])),
and facilitate the methods based on the understanding.
We find that all existing ParVIs, especially SVGD, approximate the [gradient flow](http://ml.cs.tsinghua.edu.cn/~changliu/static/Gradient-Flow.pdf)
of the KL divergence on the Wasserstein space, which drives the particle distribution towards the posterior.
The approximations of various ParVIs are essentially a smoothing operation on the particle distribution,
in either of the equivalent forms of smoothing density or smoothing functions.
This treatment is compulsory, imposing a boundary on the flexibility of ParVIs.
We develop two new ParVIs based on this finding.
Inspired by the gradient flow interpretation, we improve ParVIs by utilizing
Nesterov's acceleration method on Riemannian manifolds.
The acceleration framework can be applied to all ParVIs.
We also conceive a principled bandwidth selection method for the smoothing kernel that ParVIs use.

The repository here implements the proposed acceleration framework along with the two new ParVIs and the bandwidth selection method.
Other ParVIs ([SVGD][svgd-paper], [Blob][changyou-paper]) are also implemented.
The methods are implemented in Python with [TensorFlow][https://www.tensorflow.org/].

## Instructions
* For the synthetic experiment:

	Directly open "synthetic_run.ipynb" in a jupyter notebook.

* For the Bayesian logistic regression experiment:

	Open "blr_run.ipynb" in a jupyter notebook to run trials and view results.
	Codes are developed based on the codes of [Liu & Wang (2016)][svgd-codes].

* For the Bayesian neural network experiment:

	Edit the settings file "bnn_set_kin8nm.py" to choose a setting, and then run the command
	```bash
		python bnn_run.py bnn_set_kin8nm.py
	```
	to conduct experiment under the specified settings.
	Codes are developed based on the codes of [Liu & Wang (2016)][svgd-codes].

* For the Latent Dirichlet Allocation experiment:
	First run
	```bash
		python lda_build.py build_ext --inplace
	```
	to compile the [Cython](https://cython.org/) code, then run
	```bash
		python lda_run.py [a settings file beginning with 'lda_set_icml_']
	```
	to conduct experiment under the specified settings.

	The ICML dataset ([download here](https://cse.buffalo.edu/~changyou/code/SGNHT.zip))
	is developed and utilized by [Ding et al. (2015)](http://papers.nips.cc/paper/5592-bayesian-sampling-using-stochastic-gradient-thermostats).

	Codes are developed based on the codes of [Patterson & Teh (2013)](http://www.stats.ox.ac.uk/~teh/sgrld.html)
	for their work "[Stochastic Gradient Riemannian Langevin Dynamics for Latent Dirichlet Allocation](https://papers.nips.cc/paper/4883-stochastic-gradient-riemannian-langevin-dynamics-on-the-probability-simplex)".

## Citation
```
	@InProceedings{liu2019understanding_a,
	  title = 	 {Understanding and Accelerating Particle-Based Variational Inference},
	  author = 	 {Liu, Chang and Zhuo, Jingwei and Cheng, Pengyu and Zhang, Ruiyi and Zhu, Jun and Carin, Lawrence},
	  booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
	  pages = 	 {4082--4092},
	  year = 	 {2019},
	  editor = 	 {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
	  volume = 	 {97},
	  series = 	 {Proceedings of Machine Learning Research},
	  address = 	 {Long Beach, California USA},
	  month = 	 {09--15 Jun},
	  publisher = 	 {PMLR},
	  pdf = 	 {http://proceedings.mlr.press/v97/liu19i/liu19i.pdf},
	  url = 	 {http://proceedings.mlr.press/v97/liu19i.html},
	  organization={IMLS},
	}
```

[changliu]: http://ml.cs.tsinghua.edu.cn/~changliu/index.html
[junzhu]: http://ml.cs.tsinghua.edu.cn/~jun/index.shtml
[jingweizhuo]: http://ml.cs.tsinghua.edu.cn/~jingwei/index.html
[lcarin]: http://people.ee.duke.edu/~lcarin/
[svgd-paper]: http://papers.nips.cc/paper/6338-stein-variational-gradient-descent-a-general-purpose-bayesian-inference-algorithm
[svgd-codes]: https://github.com/DartML/Stein-Variational-Gradient-Descent
[changyou-paper]: http://auai.org/uai2018/proceedings/papers/263.pdf

