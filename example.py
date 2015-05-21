import highered

ed = highered.CRFEditDistance()

print(ed('foo', 'bar'))
print(ed('bar', 'foo'))

print(ed('foo1', 'bar'))
print(ed('bar', 'foo'1))

# X = [(u'caring hands a step ahead', u'el valor little tykes ii'),
#  (u'dulles', u"chicago public schools o'keeffe, isabell c."),
#  (u'erie neighborhood house fcch-carmen l. vega site',
#   u'erie neighborhood house fcch-servia galva site'),
#  (u'chicago public schools dvorak math & science tech academy, anton',
#   u'chicago public schools perez, manuel'),
#  (u'v & j day care center', u"henry booth house granny's day care center"),
#  (u'home of life community dev. corp. - home of life just for you',
#   u'urban family and community centers'),
#  (u'carole robertson center for learning fcch-ileana gonzalez',
#   u'carole robertson center for learning fcch-rhonda culverson'),
#  (u'bethel new life bethel child development',
#   u'mary crane league mary crane center (lake & pulaski)'),
#  (u'easter seals society of metropolitan chicago - stepping stones early/childhood lear',
#   u"marcy newberry association kenyatta's day care"),
#  (u'westside holistic family services westside holistic family services',
#   u'childserv lawndale'),
#  (u'higgins', u'higgins'),
#  (u'ymca south side', u'ymca of metropolitan chicago - south side ymca'),
#  (u'chicago commons association paulo freire',
#   u'chicago commons association paulo freire'),
#  (u'fresh start daycare, inc.',
#   u'easter seals society of metropolitan chicago fresh start day care center'),
#  (u'el valor teddy bear 3', u'teddy bear 3'),
#  (u'chicago child care society chicago child care society',
#   u'chicago child care society-child and family dev center'),
#  (u'hull house - uptown', u'uptown family care center')]
# Y = [u'distinct',
#  u'distinct',
#  u'distinct',
#  u'distinct',
#  u'distinct',
#  u'distinct',
#  u'distinct',
#  u'distinct',
#  u'distinct',
#  u'distinct',
#  u'match',
#  u'match',
#  u'match',
#  u'match',
#  u'match',
#  u'match',
#  u'match']

# ed.train(X, Y)
# print(ed.model.parameters)
# print(ed('foo', 'bar'))

# import pdb
# pdb.set_trace()
