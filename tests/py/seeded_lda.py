# import PBDef class
from pypb import PBDef
# declare distributions
# data - rows are documents as lists, each pair in list is (word_id, freq_count)
corpus = [
    [(0,4), (3,2)],
    [(2,1), (3,5)],
    [(0,4), (1,2)]]

# contains tuples with (parameter(s), 'name', number_of_cats, number_of_dice)
tokens = range(4)
topics  = range(3)
distribs = [
    (1.0, 'mu', len(topics), len(corpus)), (1.0, 'phi', len(tokens), len(topics))]

# 'generative story' for a single observation
# decorated by the PBDef class, the pb argument is the 'self' of the class 
@PBDef(distribs)
def explain(pb, doc, token, count=1):
    seeds = {0:[0], 1:[1,2]}
    for topic in seeds.get(token, topics):
        pb.draw('mu', topic, doc)
        pb.draw('phi', token, topic)
        yield

# 'generative story' for all observations
for doc,l in enumerate(corpus):
    for (token, count) in l:
        explain(doc, token, count=count)
