# import PBDef class
from pypb2 import PBDef
# import corpus
from data.arxiv_obs import corpus
import numpy as np
import dask.bag as db

# declare distributions
num_docs = 5769
num_words = 26834
num_topics = 25
distribs = [
    (1.0, 'mu', num_topics, num_docs),
    (1.0, 'phi', num_words, num_topics)
]
# 'generative story' for a single observation
# decorated by the PBDef class, the pb argument is the 'self' of the class
@PBDef(distribs)
def explain(pb, doc, token, count):
    for topic in range(num_topics):
        pb.draw('mu', topic, doc)
        pb.draw('phi', token, topic)
        yield

# 'generative story' for all observations
args = [[doc,token,count] for (doc,l) in enumerate(corpus) for (token,count) in l]

x = db.from_sequence(args)
x.map(lambda doc,token,count: explain(doc,token,count=count)).compute()

