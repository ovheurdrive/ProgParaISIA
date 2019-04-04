from time import sleep
times = [15,8,13,7,13,17,6]
philos = [0 for _ in range(7)]
baguette = [-1 for _ in range(7)]

time = 0
transition = False

print("Eating times {}".format(times))
while philos != times:
    for i in range(len(philos)):
        if philos[i] < times[i]:
            if baguette[i] == -1 and baguette[(i+1)%7] == -1:
                baguette[i] = i
                baguette[(i+1)%7] = i
                philos[i] += 1
            elif baguette[i] == i and baguette[(i+1)%7] == i:
                philos[i] += 1
    
    for i in range(len(philos)):
        if philos[i] == times[i]:
            if baguette[i] == i:
                baguette[i] = -1
            if baguette[(i+1)%7] == i:
                baguette[(i+1)%7] = -1
    print("Philosophes times: {}, baguette used by: {}, total time {}".format(philos, baguette,time))
    if baguette == [-1 for _ in range(7)] or transition:
        sleep(0.1)
        time +=0.1
    else:
        sleep(1)
        time+=1