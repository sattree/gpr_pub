from itertools import chain
import subprocess

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