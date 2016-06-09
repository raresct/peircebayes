# import PBDef class
from pypb import PBDef
# declare distributions
distribs = [(1.0, 'mu', 2, 3), (1.0, 'phi', 4, 2)]
# 'generative story' for a single observation
# decorated by the PBDef class, the pb argument is the 'self' of the class
@PBDef(distribs)
def explain(pb, doc, token, count):
    for topic in range(2):
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
