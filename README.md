# Understanding and Accelerating Particle-Based Variational Inference
## [Chang Liu](https://github.com/chang-ml-thu), Jingwei Zhuo, Pengyu Cheng, Ruiyi Zhang, Jun Zhu, and Lawrence Carin

## Instructions
* For the synthetic experiment:
	Directly open "synthetic_run.ipynb" in a jupyter notebook.

* For the Bayesian logistic regression experiment:
	Open "blr_run.ipynb" in a jupyter notebook to run trials and view results.
	Codes are developed based on the codes of "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm" (Liu and Wang, 2016).

* For the Bayesian neural network experiment:
	Edit the settings file "bnn_set_kin8nm.py" to choose a setting, and then run the command

		"python bnn_run.py bnn_set_kin8nm.py"
	
	to conduct experiment under the specified settings.
	Codes are developed based on the codes of "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm" (Liu and Wang, 2016).

* For the Latent Dirichlet Allocation experiment:
	First run
		
		"python lda_build.py build_ext --inplace"

	to compile the Cython code, then run

		"python lda_run.py [a settings file beginning with 'lda_set_icml_']"

	to conduct experiment under the specified settings.
	The ICML dataset can be downloaded from
	
		https://cse.buffalo.edu/ ̃changyou/code/SGNHT.zip

	Codes are developed based on the codes of "Stochastic Gradient Riemannian Langevin Dynamics for Latent Dirichlet Allocation" (Patterson and Teh, 2013).

