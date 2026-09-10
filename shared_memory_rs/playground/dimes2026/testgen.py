#import argparse
import sys


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def calc_hash(val:int):
    INITIAL_STATE = 0xcbf2_9ce4_8422_2325;
    PRIME = 0x0100_0000_01b3;
    
    bs=val.to_bytes(8,'little')
    h=INITIAL_STATE;
    for b in bs:
        h ^= int(b)
        h = (h * PRIME)&0xffffffff_ffffffff
    return h



def gen_distribution(size,htsize,conflicts):
    ht=[ [] for x in range(htsize)]
    mi=0
    v=0
    while(mi<conflicts) or (mi>=conflicts and v<size):
        for _ in range(htsize):
            h=calc_hash(v)%htsize
            ht[h].append(v)
            v+=1
        mi=min([len(x) for x in ht])
        print("@",v,"min:",mi)

    #ht.sort(key=lambda x: len(x),reverse=True)
    return ht



def main(argv):

    #print("hash(1)=",calc_hash(1))
    #print("hash(10)=",calc_hash(10))
    size=500
    htsize=512
    skew=0.0
    filename="out"
    probe_size=htsize*10
    probe_skew=1.0
    probe_selectivity=0.25

    print("Args:")
    print("--size=  Rows of buildtable. Default",size)
    print("--htsize=  Size of hashtable (number of buckets). Default",htsize)
    print("--skew=  0:skewed 0to 1:uniform. Default",skew)
    print("--probesize=  Rows of probetable. Default",probe_size)
    print("--probeskew=  0:skewed 0to 1:uniform. Default",probe_skew)
    print("--selectivity=  0 none: 1 all. Default",probe_selectivity)

    for arg in argv[1:]:
        if arg.startswith("--size="):
            size=int(arg.split("=")[1])
        elif arg.startswith("--htsize="):
            htsize=int(arg.split("=")[1])
        elif arg.startswith("--skew="):
            skew=float(arg.split("=")[1])
        elif arg.startswith("--filename="):
            filename=arg.split("=")[1]
        elif arg.startswith("--probesize="):
            probe_size=int(arg.split("=")[1])
        elif arg.startswith("--probeskew="):
            probe_skew=float(arg.split("=")[1])
        elif arg.startswith("--selectivity="):
            probe_selectivity=float(arg.split("=")[1])
        else:
            print("Error unkown arg",arg)
    
    

    mu, sigma = htsize/2, (skew*(htsize/2)) # mean and standard deviation
    x=np.arange(0,htsize)
    if skew>2.0:
        bins=[0]*htsize

        for i in range(size):
            bins[i%len(bins)]+=1
        bins=np.array(bins)
        
    else:
        bins=1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) )
        bins*=size
        plt.plot(x, bins , 'r.-', alpha=0.3, label='ideal count')
        bins=bins.round().astype("int64")
        compensate=size-bins.sum()
        print("to compensate:",compensate)
        while(compensate!=0):
            p=np.random.randint(0,htsize);
            if compensate>0:
                bins[p] += 1
                compensate-=1
            elif bins[p]>0:
                bins[p]-=1
                compensate+=1

    plt.plot(x, bins , 'k.-', alpha=0.3, label='expected count')



    #get related distribution
    m=bins.max()
    candidates=gen_distribution(size,htsize,m)
    m={}
    h=int(0)
    for b in bins:
        #print(b)
        m[h]=candidates[h][0:b]
        h+=1

    for i in range(htsize):
        if not i in m:
            m[i]=[]

    mi=size
    ma=0
    for v in m.values():
        l=len(v)
        mi=min(mi,l)
        ma=max(ma,l)

    print(" htusage:",sum( [1 for x in m.values() if len(x)>0]), "of",htsize)
    print("conflics:",sum([ len(x)-1 for x in m.values() if len(x)>0]))
    print("     min:",mi)
    print("     max:",ma)
    
    plt.bar(m.keys(), [len(x) for x in m.values()], label='sample count')

    s=[]
    for v in m.values():
        s.extend(v)
    np.random.shuffle(s)
    
    if not "." in filename:
        filename=filename+".dat"
    with open("build_"+filename,'w') as f:
        for v in s:
            f.write(str(v)+"\n");
    
    with open("build_frequency_"+filename,'w') as f:
        f.write("value, count\n")
        for k,v in m.items():
            f.write(str(k)+", "+ str(len(v))+"\n");
    
    plt.grid(alpha=0.4)
    plt.legend()
    plt.show()

    # probe part
    needed=int(probe_size*probe_selectivity);
    
    mu, sigma = size/2, (probe_skew*(size/2)) # mean and standard deviation
    x=np.arange(0,size)
    bins=1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) )
    bins*=needed
    plt.plot(x, bins , 'r.-', alpha=0.3, label='ideal count')
    bins=bins.round().astype("int64") 
    compensate=needed-bins.sum()
    print("to compensate:",compensate)
    while(compensate!=0):
        p=np.random.randint(0,size);
        if compensate>0:
            bins[p] += 1
            compensate-=1
        elif bins[p]>0:
            bins[p]-=1
            compensate+=1

    unused_keys_start=max(s)+1


    p=[]
    i=0
    for b in bins:
        p.extend([s[i]]*b)
        i+=1

    if len(p)<needed:
        print("Something went wrong")

    bins=np.bincount(p)
    plt.bar(np.arange(1,max(p)+1),bins[1:],label="probe hits")
    
    p.extend([x for x in range(unused_keys_start,unused_keys_start+probe_size-needed)])
    print("probe_size:",len(p), "of",probe_size)
    

    np.random.shuffle(p)
    
    with open("probe_"+filename,'w') as f:
        for v in p:
            f.write(str(v)+"\n");
    
    
    plt.grid(alpha=0.4)
    plt.legend()
    plt.show()




    


if __name__ == "__main__":
    main(sys.argv)
