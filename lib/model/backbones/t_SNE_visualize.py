import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn import manifold
import torch

def draw_tsne(RGB, T):

    '''t-SNE'''
    #RGB:[16, 256, 768]
    #T:[16, 256, 768]
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X = torch.cat([RGB, T], dim=1).detach().cpu().numpy()
    X_tsne = tsne.fit_transform(X[0].squeeze())
    print(X.shape)
    print(X_tsne.shape)
    print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    mid = int(X_norm.shape[0] / 2)
    plt.figure(figsize=(8, 8))
    for i in range(mid):
        plt.text(X_norm[i, 0], X_norm[i, 1], '*', color=plt.cm.Set1(1/10),
                 fontdict={'weight': 'bold', 'size': 18})
    for i in range(mid):
        plt.text(X_norm[mid + i, 0], X_norm[mid + i, 1], '*', color=plt.cm.Set1(2/10),
                 fontdict={'weight': 'bold', 'size': 18})
    plt.xticks([])
    plt.yticks([])
    plt.show()
