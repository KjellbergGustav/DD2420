'''

This code template belongs to
"
PGM-TUTORIAL: EVALUATION OF THE 
PGMPY MODULE FOR PYTHON ON BAYESIAN PGMS
"
Created: Summer 2017
@author: miker@kth.se

Refer to https://github.com/pgmpy/pgmpy
for the installation of the pgmpy module

See http://pgmpy.org/models.html#module-pgmpy.models.BayesianModel
for examples on fitting data

See http://pgmpy.org/inference.html
for examples on inference

'''

def separator():
    input('Enter to continue')
    print('-'*70, '\n')
    
# Generally used stuff from pgmpy and others:
import math
import random
import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination
from pgmpy.inference import BeliefPropagation
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator, K2Score, BicScore

# Specific imports for the tutorial
import pgm_tutorial_data
from pgm_tutorial_data import ratio, get_random_partition

RAW_DATA = pgm_tutorial_data.RAW_DATA
FEATURES = [f for f in RAW_DATA]

''' # Task 1 ------------ Setting up and fitting a naive Bayes PGM

data = pd.DataFrame(data=RAW_DATA)
model = BayesianModel([('delay', 'age'),
                       ('delay', 'gender'),
                       ('delay', 'avg_mat'),
                       ('delay', 'avg_cs')])
model.fit(data) # Uses the default ML-estimation

STATE_NAMES = model.cpds[0].state_names
print('State names:')
for s in STATE_NAMES:
    print(s, STATE_NAMES[s])

print('')
print(model.cpds[3])
print(ratio(RAW_DATA, lambda t: t['delay']=='0'))
print(ratio(RAW_DATA, lambda t: t['delay']=='1'))
print(ratio(RAW_DATA, lambda t: t['delay']=='>=2'))
print(ratio(RAW_DATA, lambda t: t['delay']=='NA'))

separator()

delay_0 = (ratio(RAW_DATA, lambda t: t['delay']=='0'))

delay_and_age_20 = (ratio(RAW_DATA, lambda t: t['age'] == '<=20', lambda t: t['delay']=='0'))

print("Delay 0 ", delay_0)
print("combined ", delay_and_age_20)
print("Multiplied ", delay_0*delay_and_age_20)


separator()
'''
# End of Task 1



#''' # Task 2 ------------ Probability queries (inference)

data = pd.DataFrame(data=RAW_DATA)
model = BayesianModel([('delay', 'age'),
                       ('delay', 'gender'),
                       ('delay', 'avg_mat'),
                       ('delay', 'avg_cs')])
model.fit(data) # Uses the default ML-estimation

STATE_NAMES = model.cpds[0].state_names
print('State names:')
for s in STATE_NAMES:
    print(s, STATE_NAMES[s])

# First, we have the joint P(D,)
ve = VariableElimination(model)

#2.1
q = ve.query(variables = ['delay'],evidence = {'age': '<=20'})

print("Delay given age='<=20' '\n'", q)

q = ve.query(variables = ['age'],evidence = {'delay': '0'}, \
             elimination_order = ['gender', 'avg_mat', 'avg_cs'])
print("Age given delay=0 '\n'", q)

separator()

print("You're now doing the MAP_QUERY query instead")
q = ve.map_query(variables = ['age'],evidence = {'delay': '0'})
print("Age given delay=0 '\n'", q)

#'''
# End of Task 2



''' # Task 3 ------------ Reversed PGM

data = pd.DataFrame(data=RAW_DATA)
model = BayesianModel([('age', 'delay'),
                       ('gender', 'delay'),
                       ('avg_mat', 'delay'),
                       ('avg_cs', 'delay')])
model.fit(data) # Uses the default ML-estimation

STATE_NAMES = model.cpds[0].state_names
print('State names:')
for s in STATE_NAMES:
    print(s, STATE_NAMES[s])


#delay = model.cpds[3]
ve = VariableElimination(model)
q = ve.query(variables = ['delay'])
print(q)    
vals = []
vals.append(abs(q.values[0]-ratio(RAW_DATA, lambda t: t['delay']=='0')))
vals.append(abs(q.values[1]-ratio(RAW_DATA, lambda t: t['delay']=='1')))
vals.append(abs(q.values[2]-ratio(RAW_DATA, lambda t: t['delay']=='>=2')))
vals.append(abs(q.values[3]-ratio(RAW_DATA, lambda t: t['delay']=='NA')))

print(vals)

#print(model.cpds[3].values)
zeros = 0;
for groups in delay.values:
    for cols in groups:
        for val in cols:
            for spec in val:
                for single in spec:
                    if single == 0.0:
                        zeros+=1
                    print(single, " a value ", '\n')
print(zeros)
#print(data['delay'])
#print(model.cpds[0].state_names)
#print(model.cpds[1].state_names)
#print(model.cpds[2].state_names)
#print(model.cpds[3].state_names)
separator()
'''
# End of Task 3
''' This is the student code from the discussion foruym as the other 2.7
# code didn't work as intended.
def task4():
    from scipy.stats import entropy

    data = pd.DataFrame(data=RAW_DATA)

    model1 = BayesianModel([('delay', 'age'),
                            ('delay', 'gender'),
                            ('delay', 'avg_mat'),
                            ('delay', 'avg_cs')])

    model2 = BayesianModel([('age', 'delay'),
                            ('gender', 'delay'),
                            ('avg_mat', 'delay'),
                            ('avg_cs', 'delay')])

    models = [model1, model2]

    [m.fit(data) for m in models]  # ML-fit
    S = {}
    for i in range(len(model1.cpds)):
        STATE_NAMES = model1.cpds[i].state_names
        #STATE_NAMES.add(model1.cpds[1].state_names)
        print(type(STATE_NAMES))
        print('\nState names:')
        for s in STATE_NAMES:
            print(s, STATE_NAMES[s])
            S[s] = STATE_NAMES[s]
    #S = STATE_NAMES
    print(S)
    separator()
    VARIABLES = list(S.keys())

    def random_query(variables, target):
        # Helper function, generates random evidence query
        n = random.randrange(1, len(variables) + 1)
        print(n)
        evidence = {v: random.choice(S[v]) for v in random.sample(variables, n)}
        if target in evidence: del evidence[target]
        return (target, evidence)

    queries = []
    for target in ['delay', 'age']:
        variables = [v for v in VARIABLES if v != target]
        queries.extend([random_query(variables, target) for _ in range(600)])

    divs = []
    for v, e in queries:
        # Relative frequencies, compared below
        rf = [ratio(RAW_DATA, lambda t: t[v] == s,
                    lambda t: all(t[k] == w for k, w in e.items())) \
              for s in S[v]]

        print(len(divs), '-' * 20)
        print('rf: ', rf)

        div = [(v, e), rf]
        for (i,m) in enumerate(models):
            ve = VariableElimination(m)
            q = ve.query(variables = [v], evidence = e)
            q = ve.query(variables=[v], evidence=e, show_progress=False)
            div.extend([q.values, entropy(q.values, rf)])
            print('PGM: ', i, q.values, ', Divergence:', div[-1])
        divs.append(div)

    divs2 = [r for r in divs if math.isfinite(r[3]) and math.isfinite(r[5]) and not math.isnan(r[3]) and not math.isnan(r[5])]
    # What is the meaning of what is printed below?
    #for n in [1,2,3,4]:
     #   print([len([r for r in divs2 if len(r[0][1]) == n]),
      #         len([r for r in divs2 if len(r[0][1]) == n and r[3] < r[5]]),
       #        len([r for r in divs2 if len(r[0][1]) == n and r[3] > r[5]]),
        #       len([r for r in divs if len(r[0][1]) == n and \
         #           not (math.isfinite(r[3]) and math.isfinite(r[5]))]),
          #     sum(r[3] for r in divs2 if len(r[0][1]) == n)])
    for n in [1,2,3,4]:
        m1 = len([r for r in divs2 if len(r[0][1]) == n and r[3]<r[5]])
        m2 = len([r for r in divs2 if len(r[0][1]) == n and r[3]>r[5]])
        if (m1 + m2) != 0:
            m1Wins = m1/(m1+m2)
            m2Wins = m2/(m1+m2)
        else:
            m1Wins = 0
            m2Wins = 0
        
        print([m1Wins,
               m2Wins,
               sum(r[3] for r in divs2 if len(r[0][1]) == n),
               sum(r[5] for r in divs2 if len(r[0][1]) == n),
               len([r for r in divs if len(r[0][1]) ==n and \
                    not (math.isfinite(r[3]) and math.isfinite(r[5]) and not math.isnan(r[3]) and math.isnan(r[5]))])])

    # The following is required for working with same data in next task:
    import pickle
    f = open('data.pickle', 'wb')
    pickle.dump(divs2, f)
    f.close()
task4()
        
#end of students task 4
''' 

''' # Task 4 ------------ Comparing accuracy of PGM models

from scipy.stats import entropy

data = pd.DataFrame(data=RAW_DATA)

model1 = BayesianModel([('delay', 'age'),
                       ('delay', 'gender'),
                       ('delay', 'avg_mat'),
                       ('delay', 'avg_cs')])

model2 = BayesianModel([('age', 'delay'),
                        ('gender', 'delay'),
                        ('avg_mat', 'delay'),
                        ('avg_cs', 'delay')])

models = [model1, model2]

[m.fit(data) for m in models] # ML-fit

STATE_NAMES = model1.cpds[0].state_names
print('\nState names:')
for s in STATE_NAMES:
    print(s, STATE_NAMES[s])

S = STATE_NAMES

VARIABLES = list(S.keys())

def random_query(variables, target):
    # Helper function, generates random evidence query
    n = random.randrange(1, len(variables)+1)
    evidence = {v: random.randrange(len(S[v])) for v in random.sample(variables, n)}
    if target in evidence: del evidence[target]
    return (target, evidence)

queries = []
for target in ['delay']:
    variables = [v for v in VARIABLES if v != target]
    queries.extend([random_query(variables, target) for i in range(20)])

divs = []
# divs will be filled with lists on the form
# [query, distr. in data, distr. model 1, div. model 1, distr. model 2, div. model 2]
for v, e in queries:
    try:
        # Relative frequencies, compared below
        rf = [ratio(RAW_DATA, lambda t: t[v]==s,
                    lambda t:all(t[w] == S[w][e[w]] for w in e)) \
              for s in S[v]]
        # Special treatment for missing samples
        #### if sum(rf) == 0: rf = [1/len(rf)]*len(rf) # Commented out on purpose

        print(len(divs), '-'*20)
        print('Query:', v, 'given', e)
        print('rf: ', rf)
         
        div = [(v, e), rf]
        for m in models:
            print('\nModel:', m.edges())
            ve = VariableElimination(m)
            eKeys = e.keys()
            for key in eKeys:
                e[key] = str(e[key])
            q = ve.query(variables = ['delay'],evidence = {'age': '<=20'})
            div.extend([q['delay'].values, entropy(rf, q['delay'].values)])
            print('PGM:', q[v].values, ', Divergence:', div[-1])
        divs.append(div)
    except:
        # Error occurs if variable is both target and evidence. We can ignore it.
        # (Also, this case should be avoided with current code)
        pass

divs2 = [r for r in divs if math.isfinite(r[3]) and math.isfinite(r[5])]
# What is the meaning of what is printed below?
n = 2
print([len([r for r in divs2 if len(r[0][1])==n]),
       len([r for r in divs2 if len(r[0][1])==n and r[3] < r[5]]),
       len([r for r in divs2 if len(r[0][1])==n and r[3] > r[5]]),
       len([r for r in divs if len(r[0][1])==n and \
            not(math.isfinite(r[3]) and math.isfinite(r[5]))]),
       sum(r[3] for r in divs2 if len(r[0][1])==n)])


# The following is required for working with same data in next task:
# import pickle
# f = open('data.pickle', 'wb')
# pickle.dump(divs2, f)
# f.close()

separator()

'''
# End of Task 4



''' # Task 5 ------------ Checking for overfitting

from scipy.stats import entropy

data = pd.DataFrame(data=RAW_DATA)

model1 = BayesianModel([('delay', 'age'),
                       ('delay', 'gender'),
                       ('delay', 'avg_mat'),
                       ('delay', 'avg_cs')])

model2 = BayesianModel([('age', 'delay'),
                        ('gender', 'delay'),
                        ('avg_mat', 'delay'),
                        ('avg_cs', 'delay')])

models = [model1, model2]

[m.fit(data) for m in models] # ML-fit

S = {}
for i in range(len(model1.cpds)):
    STATE_NAMES = model1.cpds[i].state_names
    #STATE_NAMES.add(model1.cpds[1].state_names)
    print(type(STATE_NAMES))
    print('\nState names:')
    for s in STATE_NAMES:
        print(s, STATE_NAMES[s])
        S[s] = STATE_NAMES[s]

"""STATE_NAMES = model1.cpds[0].state_names
print('\nState names:')
for s in STATE_NAMES:
    print(s, STATE_NAMES[s])

S = STATE_NAMES"""

# Assumes you pickled data from previous task
import pickle
divs_in = pickle.load(open('data.pickle', 'rb'))

divs = []
k_fold = 5
for k in range(k_fold):
    # Dividing data into 75% training, 25% validation.
    # Change the seed to something of your choice:
    seed = 'yuygyuuyguydrt ftygjv vj' + str(k)
    fullTrain = False
    trainSize = 0.75
    if fullTrain:
        trainSize = 1
    
    raw_data1, raw_data2 = get_random_partition(trainSize, seed)
    if fullTrain:
        raw_data2 = raw_data1
    training_data = pd.DataFrame(data=raw_data1)
    # refit with training data
    [m.remove_cpds(*m.cpds) for m in models] # Gets rid of warnings
    [m.fit(training_data) for m in models]
    for i, div in enumerate(divs_in):
        print(len(divs_in)*k + i,'/', len(divs_in)*k_fold)
        div = div[:] # Make a copy for technical reasons
        try:
            v, e = div[0]

            # Relative frequencies from validation data, compared below
            rf = [ratio(raw_data2, lambda t: t[v] == s,
                    lambda t: all(t[k] == w for k, w in e.items())) \
              for s in S[v]]
            """rf = [ratio(raw_data2, lambda t: t[v]==s,
                        lambda t:all(t[w] == S[w][e[w]] for w in e)) \
                  for s in S[v]]"""
            for m in models:
                #print('\nModel:', m.edges())
                ve = VariableElimination(m)
                test = True
                for klass, varde in e.items():
                    if klass == 'avg_cs' and varde == '2<3':
                        test = False
                if test:        
                    q = ve.query(variables = [v], evidence = e, show_progress = False)
                    div.append(entropy(q.values, rf))
                #print('PGM:', q[v].values, ', Divergence:', div[-1])
            divs.append(div)
        except IndexError:
            print('fail')

# Filter out the failures
divs2 = [d for d in divs if len(d) == 8]

# Modify the following lines according to your needs.
# Perhaps turn it into a loop as well.
for n in [1,2,3,4]:
    m1 = len([r for r in divs2 if len(r[0][1])==n and r[6] < r[7]])
    m2 = len([r for r in divs2 if len(r[0][1])==n and r[6] > r[7]])
    m1_wins = m1/(m1+m2)
    m2_wins = m2/(m1+m2)
    print(n, m1_wins,m2_wins)
separator()
'''
# End of Task 5



#''' # Task 6 ------------ Finding a better structure

data = pd.DataFrame(data=RAW_DATA)

model1 = BayesianModel([('delay', 'age'),
                       ('delay', 'gender'),
                       ('delay', 'avg_mat'),
                       ('delay', 'avg_cs')])

model2 = BayesianModel([('age', 'delay'),
                        ('gender', 'delay'),
                        ('avg_mat', 'delay'),
                        ('avg_cs', 'delay')])

models = [model1, model2]

[m.fit(data) for m in models] # ML-fit

STATE_NAMES = model1.cpds[0].state_names
print('\nState names:')
for s in STATE_NAMES:
    print(s, STATE_NAMES[s])

# Information for the curious:
# Structure-scores: http://pgmpy.org/estimators.html#structure-score
# K2-score: for instance http://www.lx.it.pt/~asmc/pub/talks/09-TA/ta_pres.pdf
# Additive smoothing and pseudocount: https://en.wikipedia.org/wiki/Additive_smoothing
# Scoring functions: https://www.cs.helsinki.fi/u/bmmalone/probabilistic-models-spring-2014/ScoringFunctions.pdf
k2 = K2Score(data)
print('Structure scores:', [k2.score(m) for m in models])

separator()

print('\n\nExhaustive structure search based on structure scores:')

from pgmpy.estimators import ExhaustiveSearch, HillClimbSearch

# Warning: Doing exhaustive search on a PGM with all 5 variables
# takes more time than you should have to wait. Hence
# re-fit the models to data where some variable(s) has been removed
# for this assignement.
raw_data2 = {'age': data['age'],
             'avg_cs': data['avg_cs'],
             'avg_mat': data['avg_mat'],
             'delay': data['delay'], # Don't comment out this one
             'gender': data['gender'],
             }

data2 = pd.DataFrame(data=raw_data2)

import time
t0 = time.time()
# Uncomment below to perform exhaustive search
#searcher = ExhaustiveSearch(data2, scoring_method=K2Score(data2))
searcher = HillClimbSearch(data2, scoring_method=BicScore(data2))
#search = searcher.all_scores()
best_model = searcher.estimate()
print(best_model.edges())
#print(best_model.scoring)
                
print('time:', time.time() - t0)

# Uncomment for printout:
#for score, model in search:
#    print("{0}        {1}".format(score, model.edges()))

separator()

#'''
# End of Task 6
