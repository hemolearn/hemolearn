Model description
=================

Blind Deconvolution Analysis (BDA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**HemoLearn** is a Python module offering a new algorithm that aims to fit a
rich multivariate decomposition of the BOLD data using a semi-blind
deconvolution and low-rank sparse decomposition. The model distinguishes two
major parts in the BOLD signal: the neurovascular coupling and the neural
activity signal.

Mathematically, if we have a single subject with :math:`P` fMRI time series of length
:math:`\widetilde{T}`, if we share the spatial maps, the considered data model is:

.. math::
	\begin{align}
		\boldsymbol{Y}_i &= \left( \sum_{m=1}^{M} \boldsymbol{\Theta}_m ^\top \boldsymbol{v}_{\delta_{im}} \right)
			~\dot{*}~ \left( \sum_{k=1}^{K} \boldsymbol{u_k}^\top \boldsymbol{z_{ik}} \right)
			+ \boldsymbol{E}_i
		\enspace .
	\end{align}

We aim to distangle the neurovascular coupling modelled by
:math:`\sum_{m=1}^{M} \boldsymbol{\Theta}_m ^\top \boldsymbol{v}_{\delta_{im}}`
from the neural activation signals modelled by
:math:`\sum_{k=1}^{K} \boldsymbol{u_k}^\top \boldsymbol{z_{ik}}` by minimizing
the following cost-function:

.. math::
	\begin{equation}
		\begin{split}
			&\min_{(\boldsymbol{U}, \boldsymbol{Z}_i, \boldsymbol{\delta}_i)} ~
			\frac{1}{2n} \sum_{i=1}^{n} \left\Vert \boldsymbol{Y}_i - \left( \sum_{m=1}^{M} \boldsymbol{\Theta}_m^\top \boldsymbol{v}_{\delta_{im}} \right) ~\dot{*}~ \left( \sum_{k=1}^{K} \boldsymbol{u}_k^\top \boldsymbol{z}_{ik} \right) \right\Vert_F^2 + \frac{1}{n} \sum_{i=1}^{n} \lambda_i \sum_{k=1}^{K} \| \nabla \boldsymbol{z}_{ik} \|_1 \enspace, \\
			&  \text{subject to} \quad \forall k \quad \|\boldsymbol{u_k}\|_1 = \eta, \quad \forall k, j \quad u_{kj} \geq 0, \quad \forall i, m \quad \delta_{im} \in [0.5, 2.0] \enspace . %\\
		\end{split}
	\end{equation}

With :math:`\lambda_i` being the temporal regularization parameter for the i-th subject, :math:`\eta` the
spatial sparcity parameter, :math:`K` the number of neural components and
:math:`M` the number of vascular regions considered.