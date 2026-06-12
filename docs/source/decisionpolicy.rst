Decision Policy
===============

A decision policy converts the continuous score produced by a ForecasterModel's belief estimator into a
discrete intervention decision. Separating belief estimation (the score) from the action (intervene or not)
lets you swap decision logic, ranging from simple thresholding to look-ahead deferral and simulation-based
voting, without retraining or modifying the underlying forecaster.

Every policy implements two methods:

* ``decide(context, score_fn)``: returns a ``(score, action, metadata)`` tuple, where ``action`` is the binary
  intervention decision and ``metadata`` is an optional dict of extra per-utterance information (e.g. the
  simulated replies used by deferral policies).
* ``fit(contexts, val_contexts, score_fn)``: tunes policy-specific parameters (such as the decision threshold)
  on a held-out validation set.

A ForecasterModel owns a single decision policy, defaulting to ``ThresholdDecisionPolicy``, and exposes it via
its ``decision_policy`` property. The policy receives the model's ``score`` function as ``score_fn`` and shares
the model's labeler and ``forecast_prob`` cache key so it can reuse already-computed forecast probabilities.

This mechanism is introduced in ["Wait! There’s a Way Out"](https://arxiv.org/abs/2605.29243).

Base Class
----------

.. automodule:: convokit.decisionpolicy.decisionPolicy
    :members:

Threshold Decision Policy
-------------------------

.. automodule:: convokit.decisionpolicy.thresholdDecisionPolicy
    :members:

Deferral Decision Policy
------------------------

.. automodule:: convokit.decisionpolicy.deferralDecisionPolicy
    :members:

Random Deferral Decision Policy
-------------------------------

.. automodule:: convokit.decisionpolicy.randomDeferralDecisionPolicy
    :members:

Simulation Average Decision Policy
----------------------------------

.. automodule:: convokit.decisionpolicy.simulationAverageDecisionPolicy
    :members:

Simulation Majority Decision Policy
-----------------------------------

.. automodule:: convokit.decisionpolicy.simulationMajorityDecisionPolicy
    :members:
