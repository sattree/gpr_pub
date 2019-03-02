from itertools import chain
import subprocess
import json

class CoreNLPServer():
    def __init__(self, classpath=None, corenlp_options=None, java_options=['-Xmx5g']):
        self.classpath = classpath
        self.corenlp_options = corenlp_options
        self.java_options = java_options
        
    def start(self):
        corenlp_options = [('-'+k, str(v)) for k,v in self.corenlp_options.items()]
        corenlp_options = list(chain(*corenlp_options))
        cmd = ['java']\
            + self.java_options \
            + ['-cp'] \
            + [self.classpath+'*'] \
            + ['edu.stanford.nlp.pipeline.StanfordCoreNLPServer'] \
            + corenlp_options
        self.popen = subprocess.Popen(cmd)
        self.url = 'http://localhost:{}/'.format(self.corenlp_options.port)
        
    def stop(self):
        self.popen.terminate()
        self.popen.wait()

# timeout is hardcoded to 60s in nltk implementation
def api_call(self, data, properties=None, timeout=240):
        default_properties = {
            'outputFormat': 'json',
            'annotators': 'tokenize,pos,lemma,ssplit,{parser_annotator}'.format(
                parser_annotator=self.parser_annotator
            ),
        }

        default_properties.update(properties or {})

        response = self.session.post(
            self.url,
            params={'properties': json.dumps(default_properties)},
            data=data.encode(self.encoding),
            timeout=timeout,
        )

        response.raise_for_status()

        return response.json()