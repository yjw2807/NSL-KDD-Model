#Everyone write a comment to make sure can access edit this thing

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from ipywidgets import interact, IntSlider, Button, HBox, VBox

np.random.seed(42)

#Data Loading
url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"

#Load Feature Names
feature_names = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes', 'land','wrong_fragment','urgent','hot',
    'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login', 'is_guest_login','count','srv_count',
    'serror_rate','srv_serror_rate', 'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count', 'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','labels'
]

#Load Dataset
dataset = pd.read_csv(url, names=feature_names)

print("Dataset shape:", dataset.shape)
print(dataset.head())
