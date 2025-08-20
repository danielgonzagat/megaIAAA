import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy
from scipy import stats

colorz = ['r', 'b', 'g', 'c', 'm', 'y', 'orange', 'k']




groupnames = glob.glob('./HDFS/ptb/results*seed0.txt')  
#groupnames = glob.glob('./HDFS/ptbprevious/results*seed0.txt')  
#groupnames = glob.glob('./HDFS/ptbold/results*.txt')  



#groupnames = glob.glob('./tmp/loss_*new*eplen_250*rngseed_0.txt')  
#groupnames = glob.glob('./tmp/loss_*new*.9_*rngseed_0.txt')  



# If you can only use 7 runs, smooth the losses within each run to obtain more reliable estimates of performance!


def mavg(x, N=20):
  return x
  #cumsum = np.cumsum(np.insert(x, 0, 0)) 
  #return (cumsum[N:] - cumsum[:-N]) / N

plt.ion()
#plt.figure(figsize=(5,4))  # Smaller figure = relative larger fonts
plt.figure()

allmedianls = []
alllosses = []
poscol = 0
minminlen = 999999
for numgroup, groupname in enumerate(groupnames):
    if "ults__" not in groupname:
        continue
    g = groupname[:-6]+"*"
    print("====", groupname)
    fnames = glob.glob(g)
    fulllosses=[]
    losses=[]
    lgts=[]
    for fn in fnames:
        if "COPY" in fn:
            continue
        if False:
            #if "seed_3" in fn:
            #    continue
            #if "seed_7" in fn:
            #    continue
            if "seed_8" in fn:
                continue
            if "seed_9" in fn:
                continue
            if "seed_10" in fn:
                continue
            if "seed_11" in fn:
                continue
            if "seed_12" in fn:
                continue
            if "seed_13" in fn:
                continue
            if "seed_14" in fn:
                continue
            if "seed_15" in fn:
                continue
        z = np.loadtxt(fn)
        
        #z = mavg(z, 10)  # For each run, we average the losses over K successive episodes

        #z = z[::10] # Decimation - speed things up!
        print(len(z))
        #if len(z) < 100:
        #    print(fn, len(z))
        #    continue
        #z = z[:90]
        lgts.append(len(z))
        fulllosses.append(z)
    minlen = min(lgts)
    if minlen < minminlen:
        minminlen = minlen
    print(minlen)
    #if minlen < 1000:
    #    continue
    for z in fulllosses:
        losses.append(z[:minlen])

    losses = np.array(losses)
    alllosses.append(losses)
    
    meanl = np.mean(losses, axis=0)
    stdl = np.std(losses, axis=0)
    cil = stdl / np.sqrt(losses.shape[0]) * 1.96  # 95% confidence interval - assuming normality
    #cil = stdl / np.sqrt(losses.shape[0]) * 2.5  # 95% confidence interval - approximated with the t-distribution for 7 d.f.

    medianl = np.median(losses, axis=0)
    allmedianls.append(medianl)
    q1l = np.percentile(losses, 25, axis=0)
    q3l = np.percentile(losses, 75, axis=0)
    
    highl = np.max(losses, axis=0)
    lowl = np.min(losses, axis=0)
    #highl = meanl+stdl
    #lowl = meanl-stdl

    xx = range(len(meanl))

    # xticks and labels
    #xt = range(0, len(meanl), 1000)
    xt = range(0, 10001, 2000)
    xtl = [str(10 * 10 * i) for i in xt]   # Because of decimation above, and only every 10th loss is recorded in the files

    #plt.plot(mavg(meanl, 100), label=g) #, color='blue')
    #plt.fill_between(xx, lowl, highl,  alpha=.2)
    #plt.fill_between(xx, q1l, q3l,  alpha=.1)
    #plt.plot(meanl) #, color='blue')
    ####plt.plot(mavg(medianl, 100), label=g) #, color='blue')  # mavg changes the number of points !
    #plt.plot(mavg(q1l, 100), label=g, alpha=.3) #, color='blue')
    #plt.plot(mavg(q3l, 100), label=g, alpha=.3) #, color='blue')
    #plt.fill_between(xx, q1l, q3l,  alpha=.2)
    #plt.plot(medianl, label=g) #, color='blue')
   
    AVGSIZE = 1
    
    xlen = len(mavg(q1l, AVGSIZE))
    #mylabel = g[g.find('type'):]
    mylabel = g
    myls = '-'
    if poscol >= len(colorz):
        myls = "--"
    plt.plot(mavg(medianl, AVGSIZE), label=mylabel, color=colorz[poscol % len(colorz)], ls=myls)  # mavg changes the number of points !
    plt.fill_between( range(xlen), mavg(q1l, AVGSIZE), mavg(q3l, AVGSIZE),  alpha=.2, color=colorz[poscol % len(colorz)])
    
    #xlen = len(mavg(meanl, AVGSIZE))
    #plt.plot(mavg(meanl, AVGSIZE), label=g, color=colorz[poscol % len(colorz)])  # mavg changes the number of points !
    #plt.fill_between( range(xlen), mavg(meanl - cil, AVGSIZE), mavg(meanl + cil, AVGSIZE),  alpha=.2, color=colorz[poscol % len(colorz)])
    
    poscol += 1
    
    #plt.fill_between( range(xlen), mavg(lowl, 100), mavg(highl, 100),  alpha=.2, color=colorz[numgroup % len(colorz)])

    #plt.plot(mavg(losses[0], 1000), label=g, color=colorz[numgroup % len(colorz)])
    #for curve in losses[1:]:
    #    plt.plot(mavg(curve, 1000), color=colorz[numgroup % len(colorz)])

ps = []
# Adapt for varying lengths across groups
#for n in range(0, alllosses[0].shape[1], 3):

#for n in range(0, minminlen):
#    ps.append(scipy.stats.ranksums(alllosses[0][:,n], alllosses[1][:,n]).pvalue)
#ps = np.array(ps)

plt.legend(loc='best', fontsize=12)
#plt.xlabel('Loss (sum square diff. b/w final output and target)')
plt.xlabel('Number of Episodes')
plt.ylabel('Loss')
#plt.xticks(xt, xtl)
#plt.tight_layout()



