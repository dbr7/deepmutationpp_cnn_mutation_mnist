import numpy as np
import pandas as pd 
import statsmodels.api as sm

from patsy import dmatrices
import statsmodels.stats.power as pw
from scipy.stats import wilcoxon
from statsmodels.tools.sm_exceptions import PerfectSeparationError

# from utils.exceptions import InvalidStatisticalTest

#calculates cohen's kappa value
def cohen_d(orig_accuracy_list, accuracy_list):
    nx = len(orig_accuracy_list)
    ny = len(accuracy_list)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.std(orig_accuracy_list, ddof=1) ** 2 + (ny-1)*np.std(accuracy_list, ddof=1) ** 2) / dof)
    result = (np.mean(orig_accuracy_list) - np.mean(accuracy_list)) / pooled_std
    return result

#calculates whether two accuracy arrays are statistically different according to GLM
def is_diff_sts_regression(orig_accuracy_list, accuracy_list, threshold = 0.05):

    p_value = p_value_glm(orig_accuracy_list, accuracy_list)
    effect_size = cohen_d(orig_accuracy_list, accuracy_list)
    is_sts = ((p_value < threshold) and effect_size <= -0.5)

    return is_sts, p_value, effect_size

def is_diff_sts_classification(orig_accuracy_list, accuracy_list, threshold = 0.05):

    p_value = p_value_glm(orig_accuracy_list, accuracy_list)
    effect_size = cohen_d(orig_accuracy_list, accuracy_list)
    is_sts = ((p_value < threshold) and effect_size >= 0.5)

    return is_sts, p_value, effect_size


def p_value_wilcoxon(orig_accuracy_list, accuracy_list):
    w, p_value_w = wilcoxon(orig_accuracy_list, accuracy_list)

    return p_value_w


def p_value_glm(orig_accuracy_list, accuracy_list):
    list_length = len(orig_accuracy_list)

    zeros_list = [0] * list_length
    ones_list = [1] * list_length
    mod_lists = zeros_list + ones_list
    acc_lists = orig_accuracy_list + accuracy_list

    data = {'Acc': acc_lists, 'Mod': mod_lists}
    df = pd.DataFrame(data)

    response, predictors = dmatrices("Acc ~ Mod", df, return_type='dataframe')
    glm = sm.GLM(response, predictors)
    glm_results = glm.fit()
    glm_sum = glm_results.summary()
    pv = str(glm_sum.tables[1][2][4])
    p_value_g = float(pv)

    return p_value_g


def power(orig_accuracy_list, mutation_accuracy_list):
    eff_size = cohen_d(orig_accuracy_list, mutation_accuracy_list)
    pow = pw.FTestAnovaPower().solve_power(effect_size=eff_size, nobs=len(orig_accuracy_list) + len(mutation_accuracy_list), alpha=0.05)
    return pow
