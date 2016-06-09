import functools
import copy

class DrawError(Exception):
    def __init__(self, a, i, k1, k2):
        self.a = a
        self.i = i
        self.k1 = k1
        self.k2 = k2
    def __str__(self):
        return ('Different draws in distribution ({}, {}):\t{} and {}'.
            format(self.a,self.i,self.k1,self.k2))

class DistribError(Exception):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return 'Invalid distribution name:\t{}'.format(self.name)

class PBDef:
    def __init__(self, distribs, out='out1.plate'):
        self.defs = {}
        self.explain = None
        self.curr_explanation = {}
        self.curr_solution = []
        self.a = 0
        self.count=1
        for distrib in distribs:
            self.add_dirichlet(*distrib)
        self.out = out
        with open(self.out, 'w') as f:
            pass

    def add_dirichlet(self, priors, name, k_a, i_a):
        self.defs[name] = (priors, self.a, k_a, i_a)
        self.a += 1

    def draw(self, name, k, i):
        if name not in self.defs:
            raise DistribError(name)
        (_, a, k_a, i_a) = self.defs[name]
        assert i<i_a
        assert k<k_a
        if (a, i) in self.curr_explanation:
            k2 = self.curr_explanation[(a,i)]
            if k != k2:
                raise DrawError(a, i, k, k2)
        else:
            self.curr_explanation[(a,i)] = k

    def __call__(self, fn):
        @functools.wraps(fn)
        def decorated(*args, **kwargs):
            self.curr_explanation = {}
            self.curr_solution = []
            for _ in fn(self, *args, **kwargs):
                self.curr_solution.append(copy.copy(self.curr_explanation))
                self.curr_explanation = {}
            self.count = kwargs['count']
            assert int(self.count) > 0
            self.write_solution()
        return decorated

    def write_solution(self):
        with open(self.out, 'a') as f:
            f.write(';'.join(
                [ '.'.join(['{},{},{}'.format(a,i,k)
                    for (a,i),k in expl.iteritems()])
                 for expl in self.curr_solution]
                 +[str(self.count)])+'\n')
