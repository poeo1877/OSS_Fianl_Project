import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import random
from sklearn import cluster
from sklearn import preprocessing


def get_n_colors(n):
    rand_colors = []
    for j in range(n):
        rand_colors.append("#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)]))
    
    return rand_colors

def get_explode(n):
    explode = []
    for j in range(n):
        explode.append(0.01)
    return explode



df = pd.read_excel('./mr-boston-flattened.xlsx', engine='openpyxl')
# https://www.kaggle.com/datasets/jenlooper/mr-boston-cocktail-dataset/code

# #다른 파일은 https://www.kaggle.com/datasets/ai-first/cocktail-ingredients에서 받아옴

# #Catagory 개수 분석 차트
# grouped_category = df.groupby('category')
# chart_data = grouped_category.size()
# fig = plt.figure(4, figsize=(3,3))
# ax = fig.add_subplot(211) 
# labels = chart_data.index
# colors = get_n_colors(len(grouped_category))
# explode = get_explode(len(grouped_category))
# wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 0}

# ax.set_title('Categorizing Cocktail Recipes')
# ax.axis("equal")
# pie = ax.pie(chart_data,autopct='%.1f%%', startangle=0, colors=colors, explode=explode, wedgeprops=wedgeprops)
# ax2 = fig.add_subplot(212)
# ax2.axis("off") 
# ax2.legend(pie[0],labels, loc="center")
# print(chart_data)
# plt.show()




# #Cocktail에서 토대가 되는 술 
# grouped_base = df.groupby('ingredient-1')
# chart_data = grouped_base.size().loc[lambda x: x > 30]    #레시피 개수가 30개 초과인 것만 filter
# print(chart_data)
# plt.title('Main ingredient')
# plt.ylabel('Number of Recipes')
# plt.bar(chart_data.index, chart_data)
# plt.show()




# # #Cocktail의 정량
# grouped_size = df.groupby('glass-size')
# chart_data = grouped_size.size().loc[lambda x: x> 30]       #레시피 개수가 30개 초과인 것만 filter
# plt.title('Cocktail glass-size')
# print(chart_data)
# plt.bar(chart_data.index, chart_data)
# plt.show()




# #Category 별 Cocktail 정량
# grouped_two = df.groupby(['category', 'glass-size'])
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# dfg = grouped_two.size().loc[lambda x: x > 5]
# colors = get_n_colors(len(dfg))
# ax = dfg.unstack(level=0).plot(kind='bar', rot=0, figsize=(9, 7), layout=(2, 4), color = colors)
# print(dfg)
# plt.show()




# #머신러닝_K-means
# ndf = df[['name', 'ingredient-1', 'ingredient-2', 'ingredient-3', 'ingredient-4', 'ingredient-5']]
# ndf.set_index('name', inplace=True)

# onehot_ingredient1 = pd.get_dummies(ndf, prefix='')
# print(ndf)
# ndf = pd.concat([ndf, onehot_ingredient1], axis=1)
# ndf.drop(['ingredient-1','ingredient-2', 'ingredient-3', 'ingredient-4', 'ingredient-5'], axis=1, inplace=True)

# # X = preprocessing.StandardScaler().fit(ndf).transform(ndf)
# kmeans = cluster.KMeans(init = "k-means++", n_clusters=15, n_init=100)
# kmeans.fit(ndf)         #모형 학습

# cluster_label = kmeans.labels_   #군집
# ndf['Cluster'] = cluster_label      #군집 결과를 DataFrame객체 열로 붙여주고...

# #군집 결과 시각화
# _ratio = ndf['Cluster'].value_counts()
# ratio = _ratio.sort_index()
# plt.pie(ratio, labels=ratio.index, autopct='%d', startangle=90, counterclock=False)
# plt.show()