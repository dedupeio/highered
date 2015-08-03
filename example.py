import csv
import random
from itertools import combinations

import highered

POSITIVE_SAMPLE = 200

names = {}
all_names = set()

addresses = {}
all_addresses = set()

with open('restaurant-nophone-training.csv') as f :
    reader = csv.DictReader(f)
    for row in reader :
        names.setdefault(row['unique_id'], []).append(row['name'])
        all_names.add(row['name'])
        addresses.setdefault(row['unique_id'], []).append(row['address'].strip())
        all_addresses.add(row['address'])

names = {k : v for k, v in names.items() if len(v) > 1}
addresses = {k : v for k, v in addresses.items() if len(v) > 1}

all_names = list(all_names)
names_1 = all_names[:]
random.shuffle(names_1)

all_addresses = list(all_addresses)
addresses_1 = all_addresses[:]
random.shuffle(addresses_1)

positive_examples = []
for entity_id in random.sample(names.keys(), POSITIVE_SAMPLE/2) :
    positive_examples.append(names[entity_id][:2])
for entity_id in random.sample(addresses.keys(), POSITIVE_SAMPLE/2) :
    positive_examples.append(addresses[entity_id][:2])

negative_examples = zip(names_1, all_names)[:POSITIVE_SAMPLE]
negative_examples += zip(addresses_1, all_addresses)[:POSITIVE_SAMPLE]

ed = highered.CRFEditDistance()

print(ed('a', 'b'))

print(ed('foo', 'bar'))
print(ed('bar', 'foo'))

print(ed('foo1', 'bar'))
print(ed('bar', 'foo1'))

X = positive_examples + negative_examples
Y = ['match'] * POSITIVE_SAMPLE + ["non-match"] * POSITIVE_SAMPLE*2

print X, Y

ed.train(X, Y)
print(ed.model.parameters)

print(ed('foo', 'bar'))

