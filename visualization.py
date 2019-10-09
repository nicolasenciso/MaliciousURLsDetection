import numpy as np
import threading
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from mpl_toolkits.mplot3d import Axes3D



#importing the dataset
#dataset = pd.read_csv('Spam_Infogain.csv')
dataset = pd.read_csv('Malware.csv')
X = dataset.iloc[:,0:-1].values
Y = dataset.iloc[:,5]


#missing data treatment
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)

#feature scaling
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

#making PCA visualization

def makePCA(X,Y):
    pca = PCA(n_components = 2)
    X_pca = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_
    count = 0
    attackX,attackY,benignX,benignY = [],[],[],[]
    for case in Y:
        if str(case) != 'benign':
            attackX.append(X_pca[count][0])
            attackY.append(X_pca[count][1])
        elif str(case) == 'benign':
            benignX.append(X_pca[count][0])
            benignY.append(X_pca[count][1])
        count = count + 1

    plt.figure()
    plt.scatter(x=benignX,y=benignY,c='blue',marker='o',label='Benign')
    plt.scatter(x=attackX,y=attackY,c='red',marker='o',label='Attack')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.title('2-PCA')
    plt.grid()
    plt.savefig('images/PCA.png')

def makeTSNE(X,Y):
    #making t-SNE visualization
    tsne = TSNE(n_components=3, random_state=0)
    x_tsne = tsne.fit_transform(X)
    count = 0
    attackX,attackY,attackZ, benignX,benignY, benignZ = [],[],[],[],[],[]
    for case in Y:
        if str(case) != 'benign':
            attackX.append(x_tsne[count][0])
            attackY.append(x_tsne[count][1])
            attackZ.append(x_tsne[count][2])
        elif str(case) == 'benign':
            benignX.append(x_tsne[count][0])
            benignY.append(x_tsne[count][1])
            benignZ.append(x_tsne[count][2])
        count = count + 1

    plt.figure()
    plt.scatter(x=benignX,y=benignY,c='blue',marker='o',label='Benign')
    plt.scatter(x=attackX,y=attackY,c='red',marker='o',label='Attack')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.title('t-SNE')
    plt.grid()
    plt.savefig('images/TSNE.png')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(attackX,attackY,attackZ, c='red',marker ='o',label='Attack')
    ax.scatter(benignX,benignY,benignZ, c='blue',marker ='o',label='Benign')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.title('t-SNE 3D')
    plt.savefig('images/TSNE3D.png')


def makeISOMAP(X,Y):
    X_std = X
    y = Y
    iso = Isomap(n_neighbors=3, n_components=2)
    x_iso = iso.fit_transform(X_std)
    count = 0
    anomalousX,anomalousY,normalX,normalY = [],[],[],[]
    for tipo in y:
        if str(tipo) != 'benign':
            anomalousX.append(x_iso[count][0])
            anomalousY.append(x_iso[count][1])
        elif str(tipo) == 'benign':
            normalX.append(x_iso[count][0])
            normalY.append(x_iso[count][1])
        count += 1

    markers=('o', 'o')
    color_map = {0:'red', 1:'blue'}
    plt.figure()
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x_iso[y==cl,0], y=x_iso[y==cl,1], c=color_map[idx], marker=markers[idx], label=cl)
    plt.xlabel('X in Isomap')
    plt.ylabel('Y in Isomap')
    plt.legend(loc='upper left')
    plt.title('Isomap visualization')
    plt.savefig('images/ISOMAP.png')


def startVisual(X,Y):
    pca = threading.Thread(target=makePCA,args=(X,Y))
    tsne = threading.Thread(target=makeTSNE,args=(X,Y))
    isomap = threading.Thread(target=makeISOMAP,args=(X,Y))
    pca.start()
    tsne.start()
    isomap.start()

#startVisual(X,Y)
#makeISOMAP(X,Y)
makePCA(X,Y)
makeTSNE(X,Y)