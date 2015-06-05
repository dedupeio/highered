import csv
import random
from itertools import combinations

import highered

POSITIVE_SAMPLE = 30

names = {}
all_names = set()

with open('restaurant-nophone-training.csv') as f :
    reader = csv.DictReader(f)
    for row in reader :
        names.setdefault(row['unique_id'], []).append(row['name'])
        all_names.add(row['name'])

names = {k : v for k, v in names.items() if len(v) > 1}

all_names = list(all_names)
names_1 = all_names[:]
random.shuffle(all_names)

positive_examples = []
for entity_id in random.sample(names.keys(), POSITIVE_SAMPLE) :
    positive_examples.append(names[entity_id][:2])

negative_examples = zip(names_1, all_names)[:POSITIVE_SAMPLE]

ed = highered.CRFEditDistance()

print(ed('a', 'b'))

print(ed('foo', 'bar'))
print(ed('bar', 'foo'))

print(ed('foo1', 'bar'))
print(ed('bar', 'foo1'))

import pdb
pdb.set_trace()

X = positive_examples + negative_examples
Y = ['match'] * POSITIVE_SAMPLE + ["distinct"] * POSITIVE_SAMPLE

print X, Y

ed.train(X, Y)
print(ed.model.parameters)

print(ed('foo', 'bar'))

