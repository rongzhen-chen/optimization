import time, random, math

import numpy as np
import matplotlib.pyplot as plt


def getminutes(t):
    x = time.strptime(t,'%H:%M')
    return x[3]*60+x[4]

def printschedule(r):
    for d in range(len(r)/2):
        name = people[d][0]
        origin = people[d][1]
        out = flights[(origin,destination)][r[2*d]]
        ret = flights[(destination,origin)][r[2*d+1]]
    #    print 'Name: {0}; Orig: {1}; Depart: from {2} to {3}; Price: {4}; Return: from {5} to {6}; Price: {7}.'.format(name, origin, out[0], out[1], out[2], ret[0], ret[1], ret[2])
        print '%10s%10s %5s-%5s $%3s %5s-%5s $%3s' % (name,origin,out[0],out[1],out[2], ret[0],ret[1],ret[2])

def schedulecost(sol):
    totalprice=0
    latestarrival=0
    earliestdep=24*60
    for d in range(len(sol)/2):
      # Get the inbound and outbound flights
      origin=people[d][1]
      outbound=flights[(origin,destination)][int(sol[2*d])]
      returnf=flights[(destination,origin)][int(sol[2*d+1])]
    
      # Total price is the price of all outbound and return flights
      totalprice+=outbound[2]
      totalprice+=returnf[2]
    
      # Track the latest arrival and earliest departure
      if latestarrival<getminutes(outbound[1]): latestarrival=getminutes(outbound[1])
      if earliestdep>getminutes(returnf[0]): earliestdep=getminutes(returnf[0])
  
    # Every person must wait at the airport until the latest person arrives.
    # They also must arrive at the same time and wait for their flights.
    totalwait=0  
    for d in range(len(sol)/2):
      origin=people[d][1]
      outbound=flights[(origin,destination)][int(sol[2*d])]
      returnf=flights[(destination,origin)][int(sol[2*d+1])]
      totalwait+=latestarrival-getminutes(outbound[1])
      totalwait+=getminutes(returnf[0])-earliestdep  

    # Does this solution require an extra day of car rental? That'll be $50!
    if latestarrival>earliestdep: totalprice+=50
  
    return totalprice+totalwait

def randomoptimize(domain,costf):
    best = 999999999
    bestr = None
    for i in range(10000):
        # creat a random solution
        r = [random.randint(domain[i][0],domain[i][1]) for i in range(len(domain))]
        cost = costf(r)
        if cost < best:
            best = cost
            bestr = r
        return r

def hillclimb(domain,costf):
    # Create a random solution
    sol=[random.randint(domain[i][0],domain[i][1]) for i in range(len(domain))]
    # Main loop
    while 1:
      # Create list of neighboring solutions
        neighbors=[]
        for j in range(len(domain)):
          # One away in each direction
            if sol[j]>domain[j][0]:
                neighbors.append(sol[0:j]+[sol[j]-1]+sol[j+1:])
            if sol[j]<domain[j][1]:
                neighbors.append(sol[0:j]+[sol[j]+1]+sol[j+1:])
        # See what the best solution amongst the neighbors is
        current=costf(sol)
        best=current
        for j in range(len(neighbors)):
            cost=costf(neighbors[j])
            if cost<best:
                best=cost
                sol=neighbors[j]

        # If there's no improvement, then we've reached the top
        if best==current:
          break
    return sol

def randomhillclimb(domain,costf,n):
    listSol=[]
    listCost=[]
    for i in range(5): 
        # Create a random solution
        sol=[random.randint(domain[i][0],domain[i][1]) for i in range(len(domain))]
        # Main loop
        while 1:
          # Create list of neighboring solutions
            neighbors=[]
            for j in range(len(domain)):
              # One away in each direction
                if sol[j]>domain[j][0]:
                    neighbors.append(sol[0:j]+[sol[j]-1]+sol[j+1:])
                if sol[j]<domain[j][1]:
                    neighbors.append(sol[0:j]+[sol[j]+1]+sol[j+1:])
            # See what the best solution amongst the neighbors is
            current=costf(sol)
            best=current
            for j in range(len(neighbors)):
                cost=costf(neighbors[j])
                if cost<best:
                    best=cost
                    sol=neighbors[j]

            # If there's no improvement, then we've reached the top
            if best==current:
              break
        listSol.append(sol) 
        listCost.append(costf(sol)) 
    print(listSol)
    print(listCost)
    indexFinal=listCost.index(min(listCost))
    print(listSol[indexFinal])
    return listSol[indexFinal]

def pltFigs():

    plt.figure(1) 
    temp=np.linspace(0.0, 1E4, num=1000)
    diffDL=100
    plt.plot(temp,np.exp(-diffDL/temp),'-o')
    plt.xlabel('Temperature')
    plt.ylabel('exp(-(highcost-lowcost)/temperature)')
    plt.title('highcost-lowcost=100 (fixed value)')
#    plt.savefig('fig1.png')

    plt.figure(2) 
    diffDL=np.linspace(-10.0, 1E4, num=1000)
    temp=100
    plt.plot(diffDL,np.exp(-diffDL/temp),'-o')
    plt.xlabel('highcost-lowcost')
    plt.ylabel('exp(-(highcost-lowcost)/temperature)')
    plt.title('temperature=100 (fixed value)')
#    plt.savefig('fig2.png')

    plt.show()

def annealingoptimize(domain,costf,T=10000.0,cool=0.95,step=1):
  # Initialize the values randomly
  vec=[(random.randint(domain[i][0],domain[i][1]))
       for i in range(len(domain))]

  while T>0.1:
    # Choose random one of the indices
    i=random.randint(0,len(domain)-1)

    # dir can be among -1, 0, 1 
    dir=random.randint(-step,step)

    # Create a new list with one of the values changed
    vecb=vec[:]
    vecb[i]+=dir
    # Control the values in the range
    if vecb[i]<domain[i][0]: vecb[i]=domain[i][0]
    elif vecb[i]>domain[i][1]: vecb[i]=domain[i][1]

    # Calculate the current cost and the new cost
    # Calculate the probability p
    ea=costf(vec)
    eb=costf(vecb)
    p=np.exp(-(eb-ea)/T)

    # Core of algorithm
    if (eb<ea or random.random()<p):
      vec=vecb

    # Decrease the temperature
    T=T*cool
  return vec

def geneticoptimize(domain, costf, popsize=50, step=1, mutprob=0.2, elite=0.2, maxiter=100):


    # two methods to contruct new vec
    def mutate(vec):
        i = random.randint(0,len(domain)-1)
        if random.random()<0.5 and vec[i]>domain[i][0]:
            return vec[0:i]+[vec[i]-step]+vec[i+1:]
        elif vec[i]<domain[i][1]:
            return vec[0:i]+[vec[i]+step]+vec[i+1:]

    def crossover(r1,r2):
        i=random.randint(1,len(domain)-2)
        return r1[0:i]+r2[i:]

    pop=[]

    for i in range(popsize):
        vec=[random.randint(domain[i][0],domain[i][1]) for i in range(len(domain))]
        pop.append(vec) 

    topelite=int(elite*popsize)

    for i in range(maxiter):
        scores=[(costf(v),v) for v in pop if v!=None]
        scores.sort()
        ranked=[v for (s,v) in scores]
        # Select the topelite number of vectors
        pop=ranked[0:topelite]

        # Reconstruct popsize number of vectors
        while len(pop)<popsize:
            if random.random()<mutprob:
                c=random.randint(0,topelite)
                pop.append(mutate(ranked[c]))
            else:
                c1=random.randint(0,topelite)
                c2=random.randint(0,topelite)
                pop.append(crossover(ranked[c1],ranked[c2]))

        #print scores[0][0]
    return scores[0][1] 


people=[('Seymour','BOS'),('Franny','DAL'),('Zooey','CAK'),('Walt','MIA'),('Buddy','ORD'),('Les','OMA')]
destination='LGA'

flights={}

for line in file('schedule.txt'):
    origin, dest, depart, arrive, price = line.strip().split(',')
    flights.setdefault((origin,dest),[])
    #save all "origin, dest" and "dest, origin"
    flights[(origin, dest)].append((depart,arrive,int(price)))



#s=[1,4,3,2,7,3,6,3,2,4,5,3]
domain=[(0,9)]*(len(people)*2)
print("domain={0}".format(domain))
#s=randomoptimize(domain,schedulecost)
print('-----------hillclimb----------')
s=hillclimb(domain,schedulecost)
print(s)
print(schedulecost(s))
print(printschedule(s))

print('-----------randomhillclimb----------')
s=randomhillclimb(domain,schedulecost,5)
print(s)
print(schedulecost(s))
print(printschedule(s))

#pltFigs()

print('-----------annealing----------')
s=annealingoptimize(domain,schedulecost,cool=0.95)
print(s)
print(schedulecost(s))
print(printschedule(s))


print('-----------genetic----------')
s=geneticoptimize(domain,schedulecost)
print(s)
print(schedulecost(s))
print(printschedule(s))
