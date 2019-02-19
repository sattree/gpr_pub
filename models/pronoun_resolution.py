import pandas as pd

from tqdm import tqdm
from sklearn.externals.joblib import Parallel, delayed

class PronounResolutionModel:
    def __init__(self, coref_model, n_jobs=1, verbose=1, backend='threading',):
        self.model = coref_model
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.backend = backend

    def batch_predict(fn):
        def _predict(self, df, **kwargs):
            if isinstance(df, pd.DataFrame):
                rows = []
                if self.n_jobs != 1:
                    with Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend=self.backend) as parallel:
                        rows = parallel([delayed(fn)(*(self, row), **kwargs) for idx, row in df.iterrows()])
                else:
                    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
                        rows.append(fn(self, row, **kwargs))
                return rows
            else:
                return fn(self, df, **kwargs)
        return _predict

    @batch_predict
    def predict(self, x, debug=False):
        tokens, clusters, pronoun_offset, a_span, b_span = self.model.predict(**x)

        pred = [False, False]
        for cluster in clusters:
            for mention in cluster:
                # if the cluster contains pronoun
                if mention[0] == pronoun_offset and mention[1] == pronoun_offset:
                    for mention in cluster:
                        # some part of token is covered as mention
                        if a_span[0] <= mention[0] and a_span[1] >= mention[1]:
                            pred = [True, False]
                        elif b_span[0] <= mention[0] and b_span[1] >= mention[1]:
                            pred = [False, True]

        if debug:
            return [pronoun_offset, pronoun_offset], a_span, b_span, tokens, clusters, pred[0], pred[1]
        
        return pred