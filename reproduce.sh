#!/usr/bin/env bash

export PYTHONPATH=./code PYTHONWARNINGS="ignore"
cd code/examples
ipython --classic --matplotlib=agg example_ConsLaborPortfolioModel.ipynb
ipython --classic --matplotlib=agg example_ConsLaborSeparableModel.ipynb
ipython --classic --matplotlib=agg example_ConsPensionModel_baseline.ipynb
ipython --classic --matplotlib=agg example_ConsPensionModel.ipynb
ipython --classic --matplotlib=agg example_ConsRetirementModel.ipynb
ipython --classic --matplotlib=agg example_GaussianProcessRegression.ipynb
ipython --classic --matplotlib=agg example_WarpedInterpolation.ipynb
