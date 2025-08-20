import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy
from scipy import stats

#colorz = ['r', 'b', 'g', 'c', 'm', 'y', 'orange', 'k']
colorz = ['r', 'm', 'b', 'c', 'y', 'orange']
#colorz = ['g', 'g', 'r', 'r', 'b', 'b']

plt.rc('font', size=14)


groupnames = glob.glob('./tmp/loss*tch*gc_4*msize_13*seed_0.txt')  
#groupnames = glob.glob('./tmp/loss*tch*gc_*msize_17*seed_0.txt')  
#groupnames = glob.glob('./tmp_prev/loss*addpw_3*md_0*msize_13*rew_1.*seed_0.txt')  
#groupnames = [x for x in groupnames if not (('pw_0' in x) or ('maz' in x) or  
#    ('modul2' in x) or ('rsp_0' in x))]  # pw_0 is bad, rsp_0 is a different setting, modul2 has similar results to to modul, maz is very slightly different

#groupnames = glob.glob('./tmp/loss*msize_11*seed_0.txt')  
#groupnames = glob.glob('./tmp/loss*l2_3e-06*md_4*msize_13*seed_0.txt')  # 11 / hs 100, modul vs modplast, with or without delay (and one w/ 4 cpus i.o. 2)
#groupnames = glob.glob('./tmp/loss*l2_3e-06*md_0*msize_13*seed_0.txt')  # 11 / hs 100, modul vs modplast, with or without delay (and one w/ 4 cpus i.o. 2)
#groupnames = glob.glob('./tmp/loss*addpw_2*l2_3e-06*md_0*msize_13*seed_0.txt')  # 11 / hs 100, modul vs modplast, with or without delay (and one w/ 4 cpus i.o. 2)



#groupnames = glob.glob('./tmp/loss*msize_15*seed_0.txt')  # 15, hs 200, modul vs modplast
#groupnames = glob.glob('./tmp/loss*hs_100*msize_13*seed_0.txt')  # 13, hs 100, modul vs modplast

#groupnames = glob.glob('./loss*seed_0.txt')  
#groupnames = glob.glob('./tmp/loss*msize_13*plastic*seed_0.txt')  
#groupnames = glob.glob('./tmp/loss*msize_9*seed_0.txt')  
#groupnames = glob.glob('./tmp/loss*modplast*seed_0.txt')  



#groupnames = glob.glob('./tmp/loss_*new*eplen_250*rngseed_0.txt')  
#groupnames = glob.glob('./loss_*rngseed_0.txt')  



# If you can only use 7 runs, smooth the losses within each run to obtain more reliable estimates of performance!


def mavg(x, N):
  cumsum = np.cumsum(np.insert(x, 0, 0)) 
  return (cumsum[N:] - cumsum[:-N]) / N

plt.ion()
#plt.figure(figsize=(5,4))  # Smaller figure = relative larger fonts
#plt.figure(figsize=(9,7))  # Smaller figure = relative larger fonts
plt.figure(figsize=(7,5))  # Smaller figure = relative larger fonts
#plt.figure()


allmedianls = []
alllosses = []
poscol = 0
maxminlen = 0
minminlen = 999999

# Generate labels, and order of curves
namez = []
for numx, x in enumerate(groupnames):
    if 'rnn' in x:
        if '139' in x:
            myname = 'Non-plastic (139 neurons)'
        else:
            myname = 'Non-plastic (100 neurons)'
    elif 'modul' in x:
        myname = "Retroactive modulation (100 neurons)"
    elif 'modplast' in x:
        myname = "Simple modulation (100 neurons)"
    elif 'plastic' in x:
        myname = "Non-modulated plasticity (101 neurons)"
    #if 'pw_3' in x:
    #    myname += " (Hard Clip)"
    #else:
    #    myname += " (Soft Clip)"
    namez.append(myname)
order = np.argsort(namez)[::-1]
namez = [namez[c] for c in order]
groupnames = [groupnames[c] for c in order]

for numgroup, groupname in enumerate(groupnames):
    if "batch"  in groupname:
        continue
    #if "lstm" not in groupname:
    #    continue
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
            if "seed_3" in fn:
                continue
            #if "seed_9" in fn:
            #    continue
            #if "seed_10" in fn:
            #    continue
            #if "seed_11" in fn:
            #    continue
            #if "seed_12" in fn:
            #    continue
            #if "seed_13" in fn:
            #    continue
            #if "seed_14" in fn:
            #    continue
            #if "seed_15" in fn:
            #    continue
        z = np.loadtxt(fn)
        
        #z = mavg(z, 10)  # For each run, we average the losses over K successive episodes

        z = z[::10] # Decimation - speed things up!

        z = z[:1000] #  Only plot the first 100K episodes (taking into account decimation above and only every 10th episode is stored in the first place)

        print(fn, len(z))
        if len(z) < 10:
            print(fn, len(z))
            continue
        #z = z[:90]
        lgts.append(len(z))
        fulllosses.append(z)
    minlen = min(lgts)
    if minlen > maxminlen:
        maxminlen = minlen
    if minlen < minminlen:
        minminlen = minlen
    print("Minlen:", minlen)
    #if minlen < 1000:
    #    continue
    for z in fulllosses:
        losses.append(z[:minlen])

    losses = np.array(losses)
    alllosses.append(losses)
    
    meanl = np.mean(losses, axis=0)
    stdl = np.std(losses, axis=0)
    #cil = stdl / np.sqrt(losses.shape[0]) * 1.96  # 95% confidence interval - assuming normality
    #cil = stdl / np.sqrt(losses.shape[0]) * 2.5  # 95% confidence interval - approximated with the t-distribution for 7 d.f.

    medianl = np.median(losses, axis=0)
    allmedianls.append(medianl)
    q1l = np.percentile(losses, 25, axis=0) # 1st quartile
    q3l = np.percentile(losses, 75, axis=0) # 3rd quartile
    
    highl = np.max(losses, axis=0)
    lowl = np.min(losses, axis=0)
    #highl = meanl+stdl
    #lowl = meanl-stdl

    xx = range(len(meanl))

    # xticks and labels
    #xt = range(0, maxminlen, 1000)
    xt = range(0, 1001, 200)
    #xt = range(0, len(meanl), 100)
    #xt = range(0, len(meanl), 1000)
    #xt = range(0, 10001, 2000)
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
   
    AVGSIZE = 20 # size of the moving average window
    
    xlen = len(mavg(q1l, AVGSIZE))
    #mylabel = g[g.find('type'):]
    mylabel = namez[numgroup]# g
    print(numgroup, mylabel)
    #if numgroup // 8 == 0:
    #    zestyle = '-'
    #elif numgroup // 8 == 1:
    #    zestyle = '--'
    #elif numgroup // 8 == 2:
    #    zestyle = ':'
    if numgroup % 2 == 0:
        zestyle = '-'
    else:
        zestyle = '--'
    
    plt.plot(mavg(medianl, AVGSIZE), label=mylabel, color=colorz[poscol % len(colorz)], ls=zestyle, lw=2)  # mavg changes the number of points !
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

a = alllosses
signifs = []
for n in range(minminlen):
    signifs.append((scipy.stats.ranksums(a[0][:,n], a[4][:,n])).pvalue)
signifs = [x[0] for x in zip(range(minlen), signifs) if x[1] < .05]

plt.plot( np.array(signifs), [20]*len(signifs), '*')

####plt.legend(loc=(.430,.15), fontsize=13)
plt.legend(loc='upper left', fontsize=13)
#plt.legend(loc='best', fontsize=13)
#plt.xlabel('Loss (sum square diff. b/w final output and target)')
plt.xlabel('Number of Episodes')
plt.ylabel('Reward')
plt.xticks(xt, xtl)
#plt.tight_layout()



