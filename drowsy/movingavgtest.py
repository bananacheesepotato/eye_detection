import time

l=[]
while True:
    time.sleep(1)
    x=time.time()
    l.append(x)
    if len(l) > 5:
        del l[0]
        print(l)
        print("average is {}".format(sum(l)/len(l)))
    