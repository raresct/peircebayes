# import PBDef class
from pypb import PBDef

# A description of the model. 
# Data consists of different threads on a forum, each thread contains posts
# and each post is authored by a user and is a bag of words
# We propose that:
# A thread is a collection of topics, each post has a latent variable corresponding
# to its topic. All words in a post are chosen from the same topic.
# So,
# There are V words in our vocabulary
# Initially K topics are created, each samples a distribution over words from
# a V dimensional dirichlet
# Each time a new thread is created, it samples a distribution over K topics
# from a K dimensional dirichlet 
# Each time a new post is created it samples a topic from the thread's distribution
# over topics
# Each word (token) in some post, i, is sampled from the kth topic, where k is the 
# unique topic of the post

# data
# threads are collections of posts
# each row is a post, posts are bags of words in pair notation, i.e. (word_id, freq_count)
#
# thread1 is mostly topic 0 and 1
thread1 = [
    [(0,4), (1,2)],
    [(2,1), (3,5)],
    [(2,4), (4,2)]]
# thread2 is mostly topic 1 and 2
thread2 = [
    [(2,4), (3,2), (4,2)],
    [(3,1), (4,2)],
    [(6,4), (7,2)]]
# thread3 is mostly topic 0 and 2
thread3 = [
    [(5,4), (7,2)],
    [(5,1), (6,2), (7,5)],
    [(0,4), (1,2)]]
# our corpus is simply our threads
threads = [
    thread1, thread2, thread3]

num_topics = 3
num_vocabulary = 8

# distribs describes all the categorical distributions in the model
# each tuple represents a group of distributions each coming from the same
# dirichlet distribution 
# contains tuples with (parameter(s), 'name', number_of_cats, number_of_dice)
distribs = [
    (1.0, 'choosetopic', num_topics, len(threads)),
    (1.0, 'choosetoken', num_vocabulary, num_topics)]

# 'generative story' for a single observation
# decorated by the PBDef class, the pb argument is the 'self' of the class 

@PBDef(distribs)
def explain(pb, threadid, post, count):
  # ???? I have an issue here I don't want to draw a new 
  # topic for each token, I only want to choose it once per post
  # I have had a guess at how to do this
  for topic in xrange(num_topics):
      pb.draw('choosetopic', topic, threadid)
      for (token, token_count) in post:
          for _ in xrange(token_count):
              pb.draw('choosetoken', token, topic)
              yield


# 'generative story' for all observations
for threadid, thread in enumerate(threads):
    for post in thread:
        explain(threadid, post, count=1)

