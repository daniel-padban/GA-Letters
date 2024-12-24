from unittest import result
import numpy as np
import pandas as pd
import statsmodels.api as sm
from results import resultsMaker



def results_to_df(result_list:list,sizes:list):
    result_dfs = []
    for i,dataset in enumerate(result_list):
        dataset_res = pd.DataFrame(dataset)
        dataset_size = sizes[i]
        dataset_res['size'] = dataset_size
        result_dfs.append(dataset_res)
    results = pd.concat(result_dfs)

    return results



results_creator = resultsMaker('B')
test_results, train_results, sizes = results_creator.make_lists()
results_to_df(test_results[0],sizes)


