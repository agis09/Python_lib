import numpy as np
import seaborn
import matplotlib.pyplot as plt
import pandas as pd

def confusion_matrix(label_list,preds,labels,save_dir,fig_name='conf_matrix.png',figsize=(10,7)):
    heat_map=np.zeros((labels.shape[1],labels.shape[1]))
    for i, j in zip(preds,labels):
        heat_map[np.where(j==1)]+=i
    heat_map/=preds.shape[0]
    df_cm=pd.DataFrame(heat_map,index=['true_'+i for i in label_list],columns=['pred_'+i for i in label_list])
    plt.figure(figsize=figsize)
    seaborn.heatmap(df_cm,annot=True)
    plt.savefig(save_dir+fig_name)



"""
### How to use:

label_list=['a','b','c']
y_pred=np.array([[0,0,1],
                 [0,1,0],
                 [1,0,0]])

y_val=np.array( [[0,0,1],
                 [0,1,0],
                 [1,0,0]])
save_dir='./'

confusion_matrix(label_list,y_pred,y_val,save_dir)
"""