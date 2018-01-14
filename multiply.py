import random

trainfile = open('multiply.csv', 'w')

N = 10000
for i in range(N):
    a = round(random.random(), 2)
    b = round(random.random(), 2)
    ans = round(a*b, 2)
    outputline = '{0},{1},{2}\n'.format(a,b,ans)
    trainfile.write(outputline)

trainfile.close()
