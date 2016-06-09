# import PBDef class
from pypb2 import PBDef
import dask.bag as db
# declare distributions
num_topics = 2
num_docs = 3
num_words = 4
distribs = [(1.0, 'mu', num_topics, num_docs), (1.0, 'phi', num_words, num_topics)]
# 'generative story' for a single observation
# decorated by the PBDef class, the pb argument is the 'self' of the class
@PBDef(distribs)
def explain(pb, doc, token, count):
    for topic in range(num_topics):
        pb.draw('mu', topic, doc)
        pb.draw('phi', token, topic)
        yield
# data
corpus = [
    [(0,4), (3,2)],
    [(2,1), (3,5)],
    [(0,4), (1,2)]
]
# 'generative story' for all observations
for doc,l in enumerate(corpus):
    for (token, count) in l:
        explain(doc, token, count=1)

args = [[doc,token,count] for (doc,l) in enumerate(corpus) for (token,count) in l]

#x = db.from_sequence(args, npartitions=100)
#x.map(lambda doc,token,count: explain(doc,token,count=count)).compute()
#for doc,token,count in args:
#    explain(doc,token,count=count)
#explain(0,0,count=4)

#explain.infer(20)
