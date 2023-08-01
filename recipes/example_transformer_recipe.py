from h2oaicore.systemutils import print_debug
import logging
from h2oaicore.systemutils import config
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning


from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
import pandas as pd
import logging

from h2oaicore.systemutils import config
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning

class ExampleLogTransformer(CustomTransformer):
    _regression = True
    _binary = True
    _multiclass = True
    _numeric_output = True
    _is_reproducible = True
    _excluded_model_classes = ['tensorflow']
    _modules_needed_by_name = ["lifelines==0.27.7"] # Not really needed, added as demo

    @staticmethod
    def do_acceptance_test():
        return True

    @property
    def logger(self):
        from h2oaicore import application_context
        from h2oaicore.systemutils import exp_dir
        # Don't assign to self, not picklable
        return make_experiment_logger(experiment_id=application_context.context.experiment_id, tmp_dir=None,
                                      experiment_tmp_dir=exp_dir())

    @staticmethod
    def get_default_properties():
        return dict(col_type = "numeric", min_cols = 1, max_cols = 1, relative_importance = 1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        logger = self.logger
        loggerinfo(logger, "Start Example Transformer fit_transform .....")
        try:
            X_pandas = X.to_pandas()
            X_p_log = np.log10(X_pandas)
        except Exception as e:
            '''Print error message into DAI log file'''
            loggerinfo(logger, 'Error during Example transformer fit_transform. Exception raised: %s' % str(e))
            raise
        return X_p_log


    def transform(self, X: dt.Frame, y: np.array = None):
        logger = self.logger
        loggerinfo(logger, "Start Example Transformer transform.......")
        try:
            X_pandas = X.to_pandas()
            X_p_log = np.log10(X_pandas)
        except Exception as e:
            '''Print error message into DAI log file'''
            loggerinfo(logger, 'Error during Example transformer transform. Exception raised: %s' % str(e))
            raise
        return X_p_log