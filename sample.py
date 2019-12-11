
import random
import pandas as pd
import numpy as np

random.seed(0)
np.random.seed(0)


class DFSampler:

    def __init__(self, df):
        ## df: pandas df containing all data elements we want to sample from
        self.df = df

    def sample_all(self, df_ratio, sample_column=None, 
            replacement=False, return_sorted=True):
        ## Returns an index list of samples.
        ##   df_ratio:  [0,1] percent of entire 
        ##   sample_column: if not set, then randomly samples over entire df.
        ##                  otherwise, maintains ratios for the class given
        ##   replacement: sample with or without replacement
        assert df_ratio > 0. and df_ratio < 1.
        if sample_column:
            assert sample_column in self.df.columns.value

        N_sample = int(df_ratio * len(self.df))
        idx_list = range(len(self.df))

        sampled_idxs = []
        if sample_column:
            col_values = list(set(self.df[sample_column].to_list()))
            cls_idxs = [[] for _ in range(len(col_values))]
            for i in range(len(self.df)):
                cidx = col_values.index(self.df.iloc[i, sample_column])
                cls_idxs[cidx].append(i)

            for cidxs in cls_idxs:
                target_count = int(len(cidxs)*df_ratio)
                sampled_idxs += list( np.random.choice(cidxs, 
                                                  size=(target_count), 
                                                  replace=replacement) )

        else:  # no class ratio restraints
            sampled_idxs = list( np.random.choice(idx_list, 
                                                  size=(N_sample), 
                                                  replace=replacement) )

        if return_sorted:
            return sorted(list(sampled_idxs))
        return list(sampled_idxs)



### Testing ###

if __name__ == '__main__':
    from collections import OrderedDict
    dfdict = OrderedDict( [('id', range(20)), \
                           ('label', [0]*10 + [1]*10)] )
    df = pd.DataFrame(dfdict)
    dfs = DFSampler(df)

    print(dfs.sample_all(0.5, replacement=False))
    print(dfs.sample_all(0.5, replacement=True))
    


        
