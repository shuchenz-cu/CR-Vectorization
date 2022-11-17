def get_rand_size(lb:int, up:int):
    from random import randint
    return randint(lb, up)

def train_knn(n:int, metric:str, data):
    import time

    from sklearn.neighbors import NearestNeighbors
    start = time.time()
    model = NearestNeighbors(n_neighbors=n,
                            metric=metric,
                            algorithm='brute')
    model.fit(data)
    running_time = time.time() - start
    return model, running_time

def get_dist_idx(model, input_id, data):
    distances,indices = model.kneighbors(data.iloc[input_id,:].values.reshape(1,-1))
    return distances, indices

# Please only input the url column as data
def get_recommend_url(distance, indices, index, data):
    input_url = [index, data[index]]
    rec_url = []
    for i in range(1,len(distance.flatten())):
        idx = indices.flatten()[i]
        url = data[idx]
        rec_url.append([idx,url])
    return input_url, rec_url

def hit_rate(input, rec, data, col):
    import numpy as np
    import pandas as pd
    input_id = input[0]
    rec = pd.DataFrame(rec, columns=['id', 'url'])
    rec_id = rec['id']
    input_cate = data.iloc[input_id, :][col]
    rec_cate = data.iloc[rec_id, :][col]
    result = rec_cate.str.match(pat = input_cate)
    return np.nansum(result)/len(rec_cate)

def display(url_list):
    from IPython.display import display, Markdown
    import ipyplot
    ipyplot.plot_images(url_list, max_images = 100, img_width=150)
    print()