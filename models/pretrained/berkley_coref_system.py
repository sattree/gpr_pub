import os
import subprocess

from tqdm import tqdm
from collections import defaultdict
from allennlp.data.dataset_readers.dataset_utils.ontonotes import Ontonotes

from ..heuristics.coref import Coref
from ..heuristics.stanford_base import StanfordModel

class BCS(Coref, StanfordModel):
    def __init__(self, model):
        self.model = model
        super().__init__(model)

    @staticmethod
    def preprocess(df):
        if not os.path.exists('tmp/text'):
            os.makedirs('tmp/text')
            os.makedirs('tmp/preprocessed')
            os.makedirs('tmp/coref')

        for idx, row in tqdm(df.iterrows()):
            with open('tmp/text/{0}'.format(row.id), 'w', encoding='utf-8') as f:
                f.write(row.text)

        print('Running BCS preprocessor')
        subprocess.run('cd berkeley-entity && java -Xmx1g -cp berkeley-entity-1.0.jar edu.berkeley.nlp.entity.preprocess.PreprocessingDriver ++config/base.conf -execDir ../tmp/logs -inputDir ../tmp/text -outputDir ../tmp/preprocessed', shell=True, stdout=None, stderr=None)
        print('Running BCS coref')
        subprocess.run('cd berkeley-entity && java -Xmx1g -cp berkeley-entity-1.0.jar edu.berkeley.nlp.entity.Driver ++config/base.conf -execDir ../tmp/logs/ -mode COREF_PREDICT -modelPath models/coref-onto.ser.gz -testPath ../tmp/preprocessed/ -outputPath ../tmp/coref -corefDocSuffix ""', shell=True, stdout=None, stderr=None)
        
        # os.remove('tmp/text/{}'.format(id))
        # os.remove('tmp/preprocessed/{}'.format(id))
        # os.remove('tmp/coref/{}-0.pred_conll'.format(id))
        
    def predict(self, text, a, b, pronoun_offset, a_offset, b_offset, id=None, debug=False, **kwargs):
        doc, tokens_, pronoun_offset, a_offset, b_offset, a_span, b_span, pronoun_token, a_tokens, b_tokens = self.tokenize(text, 
                                                                                                        a, 
                                                                                                        b, 
                                                                                                        pronoun_offset, 
                                                                                                        a_offset, 
                                                                                                        b_offset, 
                                                                                                        **kwargs)
        
        data = Ontonotes().dataset_document_iterator('tmp/coref/{}-0.pred_conll'.format(id))
        for i, doc in enumerate(data):
            tokens = []
            clusters = defaultdict(list)
            for fi in doc:
                for c in fi.coref_spans:
                    clusters[c[0]].append([len(tokens)+c[1][0], len(tokens)+c[1][1]])
                tokens += fi.words
                
        tokens = [token.replace('\\*', '*').replace('-LRB-', '(').replace('-RRB-', ')') for token in tokens]
        
        if any([token not in tokens for token in tokens_[a_span[0]:a_span[1]]+tokens_[b_span[0]:b_span[1]]]):
            print('Tokens dont match', tokens, tokens_, a, b)

        return tokens, list(clusters.values()), pronoun_offset, a_span, b_span