Model description
=================

Sparse Low-Rank Deconvolution Analysis (SLRDA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**HemoLearn** is a Python module offering a new algorithm that aims to fit a
rich multivariate decomposition of the BOLD data using a semi-blind
deconvolution and low-rank sparse decomposition. The model distinguishes two
major parts in the BOLD signal: the neurovascular coupling and the neural
activity signal.

Mathematically, if we have :math:`P` fMRI time series of length
:math:`\widetilde{T}`, the considered data model is:

.. math::
	\begin{align}
		\boldsymbol{Y} &= \left( \sum_{m=1}^{M} \boldsymbol{\Theta}_m ^\top \boldsymbol{v}_{\delta_m} \right)
			~\dot{*}~ \left( \sum_{k=1}^{K} \boldsymbol{u_k}^\top \boldsymbol{z_k} \right)
			+ \boldsymbol{E}
		\enspace .
	\end{align}

We aim to distangle the neurovascular coupling modelled by
:math:`\sum_{m=1}^{M} \boldsymbol{\Theta}_m ^\top \boldsymbol{v}_{\delta_m}`
from the neural activation signals modelled by
:math:`\sum_{k=1}^{K} \boldsymbol{u_k}^\top \boldsymbol{z_k}` by minimizing
the following cost-function:

.. math::
	\begin{equation}
		\begin{split}
			&\min_{(\boldsymbol{U}, \boldsymbol{Z}, \boldsymbol{\delta})} ~
			\frac{1}{2} \left\Vert \boldsymbol{Y} - \left( \sum_{m=1}^{M} \boldsymbol{\Theta}_m^\top \boldsymbol{v}_{\delta_m} \right) ~\dot{*}~ \left( \sum_{k=1}^{K} \boldsymbol{u}_k^\top \boldsymbol{z}_k \right) \right\Vert_F^2 + \lambda \sum_{k=1}^{K} \| \nabla \boldsymbol{z}_k \|_1 \enspace, \\
			&  \text{subject to} \quad \forall k, \|\boldsymbol{u_k}\|_1 = \eta, \quad \forall j, u_{kj} \geq 0, \quad \forall m, \delta_m \in [0.5, 2.0] \enspace . %\\
		\end{split}
	\end{equation}

With :math:`\lambda` being the temporal regularization parameter, :math:`\eta` the
spatial sparcity parameter, :math:`K` the number of neural components and
:math:`M` the number of vascular regions considered.