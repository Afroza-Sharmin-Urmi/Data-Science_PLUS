#!/usr/bin/python3

from sys         import argv
from math        import log, log2
from scipy.stats import chi2
from collections import Counter
from rules       import show, covers, colmap, prepare, allcons

#-----------------------------------------------------------------------

def clsdist (data, cid=-1):
    '''Determine the class frequency distribution in a given data set.
    data    list of data tuples, each a list of attribute values
    cid     index of the class column (default: last column)
    returns the class frequency distribution as a dictionary'''
    dist = Counter()            # count class frequencies
    for d in data: dist[d[cid]] += 1
    return dist                 # return class frequency distribution

#-----------------------------------------------------------------------

def entropy (dist):
    '''Compute Shannon entropy of a discrete probability distribution.
    dist    dictionary mapping discrete values to value frequencies
    returns Shannon entropy of the given distribution'''
    if len(dist) <= 1: return 0 # only one value: entropy is zero
    h = -sum(k *log2(k) for k in dist.values() if k > 0)
    s = sum(dist.values())      # sum entropy terms and frequencies
    return max(0, h/s +log2(s)) # compute and return Shannon entropy

#-----------------------------------------------------------------------

def pvalue (dobs, dexp):
    '''Compute p-value with likelihood ratio statistic / G statistic.
    dobs    observed distribution as a dictionary value -> frequency
    dexp    expected distribution as a dictionary value -> frequency
    returns the p-value of the observed frequency distribution'''
    r = sum(dobs.values())/sum(dexp.values())
    x = 2*sum(dobs[v] *log(dobs[v]/(r*dexp[v])) for v in dobs)
    return chi2.sf(x, len(dexp)-1)   # compute p-value from chi^2 dist.

#-----------------------------------------------------------------------

def specialize (cons, call):
    '''Specialize a list of conditions with all conditions in a set.
    cons    set of conditions to specialize
    call    set of all conditions that may be used to specialize
    returns a set of specialized conditions'''
    tabu = {c[:2] for c in cons}     # collect non-excluded conditions
    call = {frozenset([c]) for c in call if c[:2] not in tabu}
    return {cons | c for c in call}  # specialize set of conditions

#-----------------------------------------------------------------------

def bestrule (data, call, cls, bmax=5, siglvl=0.1):
    '''Find the best rule for given data.
    data    list of data tuples, each a list of attribute values
    call    list of all conditions that may be used in a rule antecedent
    cls     class information as a pair (class column index, class name)
    bmax    maximum number of rule antecedents for the beam search
    siglvl  significance level for the rule selection
    returns the best rule found'''
    dexp = clsdist(data,cls[0]) # get (expected) class distribution
    star = {frozenset()}        # initialize search set and best rule
    best = [[],[True]*len(data),dexp,entropy(dexp),0.0]
    while star:                 # while beam is not empty
        snew = {t for s in star for t in specialize(s,call)} -star
        cind = [(s,[covers(s,d) for d in data]) for s in snew]
        cvrd = [(s,x,[d for d,c in zip(data,x) if c]) for s,x in cind]
        cdst = [(s,x,clsdist(d,cls[0])) for s,x,d in cvrd if d]
        if not cdst: break      # check for applicable rules
        cval = [(s,x,d,entropy(d),pvalue(d,dexp)) for s,x,d in cdst]
        cval = [c for c in cval if c[-1] <= siglvl]
        if not cval: break      # filter with significance level
        cval.sort(key=lambda x: x[3:])
        bnew = cval[0]          # sort rules and get best current rule
        if bnew[3] < best[3]:   # if a better rule has been found,
            best = bnew         # replace the currently best rule
        star = {c[0] for c in cval[:bmax]} # limit number of rules
        # The number of rules to specialize is limited (beam search).
    c = max(best[2].items(),key=lambda x: x[1])[0]
    c = cls+('=',c)             # form rule consequent (majority class)
    return list(best[0])+[c],best[1],best[2]  # return best rule found

#-----------------------------------------------------------------------

def cn2 (data, call, cls, bmax=5, siglvl=0.1):
    '''Find rules with the CN2 algorithm [Clark & Niblett 1989].
    data    list of data tuples, each a list of attribute values
    call    all possible conditions for rule antecedents
    cls     tuple of column index and class attribute name
    returns a list of induced classification rules'''
    rules = []                  # initialize the rule set
    call  = set(call)           # and turn conditions into a set
    while data:                 # while there are uncovered cases
        print('[%d remaining]' % len(data))
        rule,cind,dist = bestrule(data, call, cls, bmax, siglvl)
        show(rule)              # generate next rule and show it
        print(' '.join('%s:%d' % d for d in dist.items()))
        rules.append(rule)      # append rule to result list
        data = [d for d,c in zip(data,cind) if not c]
        if data: print()        # reduce to uncovered cases
    return rules                # return found set of rules

#-----------------------------------------------------------------------

if __name__ == '__main__':
    fname  =       argv[1]  if len(argv) > 1 else 'iris.tab'
    bmax   =   int(argv[2]) if len(argv) > 2 else 5
    siglvl = float(argv[3]) if len(argv) > 3 else 0.1
    with open(fname, 'r') as inp:
        data = [line.rstrip('\r\n').split('\t') for line in inp]
    hdr,data = data[0],data[1:] # read data file and get header
    cid   = len(hdr)-1          # get class column index (last column)
    cmap  = colmap(hdr, data)   # create column to att. info. map
    data  = prepare(data, cmap) # prepare data (imputation, type conv.)
    call  = allcons(cmap, cid)  # generate all possible conditions
    rules = cn2(data,call,(cid,hdr[cid]),bmax,siglvl) # find rules

