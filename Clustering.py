import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs
import sys

if __name__ == "__main__":
    # number of data points in each cluster
    samp=int(500)

    #number of cluster centers 
    cents=int(2)

    #cluster standard deviation
    cstd=.2

    # takes user provided info to alter cluster parameters
    if '-samp' in sys.argv:
        p = sys.argv.index('-samp')
        stemp = float(sys.argv[p+1])
        if stemp > 0:
            samp = int(stemp)
    
    if '-cents' in sys.argv:
        p = sys.argv.index('-cents')
        centa= float(sys.argv[p+1])
        if centa > 0:
            cents = int(centa)

    if '-cstd' in sys.argv:
        p = sys.argv.index('-cstd')
        cs = float(sys.argv[p+1])
        if cs > 0:
            cstd = cs

    # create simulated clusters using scikit learn's make_blobs
    data, true_cluster = make_blobs(n_samples=samp, 
                                    centers=cents,
                                    random_state=0, 
                                    cluster_std=cstd)
    data_df = pd.DataFrame(data)
    data_df.columns=['x','y']
    data_df['true_cluster'] = true_cluster
    data_df.head(n=3)
    color_map= {0:'red',1:'yellow',2:'green'}
    data_df['true_color'] = data_df.true_cluster.map(color_map)
    data_df.head(n=3)
    
    #makes plot of cluster data
    plt.scatter(x='x',y='y',c='true_color',data=data_df)
    plt.xlabel("x")
    plt.xlabel("y")
    plt.show()
