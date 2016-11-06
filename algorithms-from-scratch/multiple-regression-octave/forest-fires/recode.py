
months = {
  'jan': '1',
  'feb': '2',
  'mar': '3',
  'apr': '4',
  'may': '5',
  'jun': '6',
  'jul': '7',
  'aug': '8',
  'sep': '9',
  'oct': '10',
  'nov': '11',
  'dec': '12'
}

days = {
  'mon': '1', 
  'tue': '2', 
  'wed': '3', 
  'thu': '4', 
  'fri': '5', 
  'sat': '6', 
  'sun': '7'
}

f1 = open('data.csv', 'r')
f2 = open('data-clean.csv', 'w')
next(f1)
for line in f1:
  vals = line.split(",")
  vals[2] = months[vals[2]]
  vals[3] = days[vals[3]]
  f2.write(','.join(vals))

