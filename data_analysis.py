import pandas as pd
import numpy as np
from collections import defaultdict

movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

# 分析电影类型分布
print("———————— 电影类型分布分析 ————————")
# 处理类型字段，可能包含多个类型用"/"分隔
genres_list = []
for genres in movies_df['GENRES'].dropna():
    genres_list.extend(genres.split('/'))

genres_counts = pd.Series(genres_list).value_counts()
print("电影类型分布（前10）:")
print(genres_counts.head(10))

# 分析电影评分分布
print("\n———————— 电影评分分布分析 ————————")
# 使用豆瓣评分
movie_ratings = movies_df['DOUBAN_SCORE'].dropna()
print(f"有评分的电影数量: {len(movie_ratings)}")
print("豆瓣评分分布:")
print(movie_ratings.describe())

# 评分区间分布
rating_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rating_labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10']
movie_rating_dist = pd.cut(movie_ratings, bins=rating_bins, labels=rating_labels, include_lowest=True)
print("\n电影评分区间分布:")
print(movie_rating_dist.value_counts().sort_index())

print("\n———————— 用户评分分布分析 ————————")
# 用户评分分布
user_rating_dist = ratings_df['RATING'].value_counts().sort_index()
print("用户评分分布:")
print(user_rating_dist)

# 用户评分统计
print("\n用户评分统计:")
print(ratings_df['RATING'].describe())

# 每个用户的评分数量分布
user_rating_counts = ratings_df.groupby('USER_MD5')['RATING'].count()
print(f"\n参与评分的用户数: {len(user_rating_counts)}")
print("用户评分数量分布:")
print(user_rating_counts.describe())

print("\n———————— 电影类型与评分关系分析 ————————")
# 计算每种类型的平均评分
genre_ratings = {}
for idx, row in movies_df.iterrows():
    if pd.notna(row['GENRES']) and pd.notna(row['DOUBAN_SCORE']):
        genres = row['GENRES'].split('/')
        for genre in genres:
            if genre not in genre_ratings:
                genre_ratings[genre] = []
            genre_ratings[genre].append(row['DOUBAN_SCORE'])

# 计算每种类型的平均评分和电影数量
genre_avg_ratings = {}
for genre, ratings in genre_ratings.items():
    genre_avg_ratings[genre] = {
        'avg_rating': np.mean(ratings),
        'count': len(ratings)
    }

# 转换为DataFrame并排序
genre_rating_df = pd.DataFrame.from_dict(genre_avg_ratings, orient='index')
genre_rating_df = genre_rating_df.sort_values('avg_rating', ascending=False)

print("各类型电影平均评分（按评分降序）:")
print(genre_rating_df.head(15))

print("\n———————— 导演与作品评分关系分析 ————————")
# 分析导演与评分关系（至少有3部作品的导演）
director_ratings = {}
for idx, row in movies_df.iterrows():
    if pd.notna(row['DIRECTORS']) and pd.notna(row['DOUBAN_SCORE']):
        directors = row['DIRECTORS'].split('/')
        for director in directors:
            if director not in director_ratings:
                director_ratings[director] = []
            director_ratings[director].append(row['DOUBAN_SCORE'])

# 筛选有至少3部作品的导演
director_avg_ratings = {}
for director, ratings in director_ratings.items():
    if len(ratings) >= 3:
        director_avg_ratings[director] = {
            'avg_rating': np.mean(ratings),
            'count': len(ratings)
        }

# 转换为DataFrame并排序
director_rating_df = pd.DataFrame.from_dict(director_avg_ratings, orient='index')
director_rating_df = director_rating_df.sort_values('avg_rating', ascending=False)

print("导演平均评分（至少3部作品，按评分降序）:")
print(director_rating_df.head(10))

print("\n———————— 演员与作品评分关系分析 ————————")
# 分析演员与评分关系（至少有3部作品的演员）
actor_ratings = {}
for idx, row in movies_df.iterrows():
    if pd.notna(row['ACTORS']) and pd.notna(row['DOUBAN_SCORE']):
        actors = row['ACTORS'].split('/')
        for actor in actors:
            if actor not in actor_ratings:
                actor_ratings[actor] = []
            actor_ratings[actor].append(row['DOUBAN_SCORE'])

# 筛选有至少3部作品的演员
actor_avg_ratings = {}
for actor, ratings in actor_ratings.items():
    if len(ratings) >= 3:
        actor_avg_ratings[actor] = {
            'avg_rating': np.mean(ratings),
            'count': len(ratings)
        }

# 转换为DataFrame并排序
actor_rating_df = pd.DataFrame.from_dict(actor_avg_ratings, orient='index')
actor_rating_df = actor_rating_df.sort_values('avg_rating', ascending=False)

print("演员平均评分（至少3部作品，按评分降序）:")
print(actor_rating_df.head(10))

print("\n———————— 演员-导演合作网络分析 ————————")
# 构建演员-导演合作网络
cooperation_network = defaultdict(lambda: defaultdict(int))

for idx, row in movies_df.iterrows():
    if pd.notna(row['ACTORS']) and pd.notna(row['DIRECTORS']):
        actors = row['ACTORS'].split('/')
        directors = row['DIRECTORS'].split('/')

        for actor in actors:
            for director in directors:
                cooperation_network[actor][director] += 1

# 转换为边列表格式
edges = []
for actor, directors in cooperation_network.items():
    for director, count in directors.items():
        if count >= 2:  # 只保留合作2次以上的关系
            edges.append({
                'actor': actor,
                'director': director,
                'count': count
            })

print(f"演员-导演合作关系数量（合作2次以上）: {len(edges)}")
print("前10个合作关系:")
for edge in sorted(edges, key=lambda x: x['count'], reverse=True)[:10]:
    print(f"{edge['actor']} - {edge['director']}: {edge['count']}次合作")


# 准备可视化数据
visualization_data = {}

# 电影类型分布数据（前10）
genres_list = []
for genres in movies_df['GENRES'].dropna():
    genres_list.extend(genres.split('/'))
genres_counts = pd.Series(genres_list).value_counts().head(10)
visualization_data['genreDistribution'] = {
    'labels': genres_counts.index.tolist(),
    'data': genres_counts.values.tolist()
}

# 电影评分分布数据
movie_ratings = movies_df['DOUBAN_SCORE'].dropna()
rating_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rating_labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10']
movie_rating_dist = pd.cut(movie_ratings, bins=rating_bins, labels=rating_labels, include_lowest=True)
movie_rating_counts = movie_rating_dist.value_counts().sort_index()
visualization_data['movieRatingDistribution'] = {
    'labels': movie_rating_counts.index.tolist(),
    'data': movie_rating_counts.values.tolist()
}

# 用户评分分布数据
user_rating_dist = ratings_df['RATING'].value_counts().sort_index()
visualization_data['userRatingDistribution'] = {
    'labels': [f'{int(label)}分' for label in user_rating_dist.index.tolist()],
    'data': user_rating_dist.values.tolist()
}

# 电影类型与评分关系数据（前15个类型）
genre_ratings = {}
for idx, row in movies_df.iterrows():
    if pd.notna(row['GENRES']) and pd.notna(row['DOUBAN_SCORE']) and pd.notna(row['DOUBAN_VOTES']):
        genres = row['GENRES'].split('/')
        for genre in genres:
            if genre not in genre_ratings:
                genre_ratings[genre] = {
                    'ratings': [],
                    'votes': 0  # 累计评分数
                }
            genre_ratings[genre]['ratings'].append(row['DOUBAN_SCORE'])
            genre_ratings[genre]['votes'] += int(row['DOUBAN_VOTES'])  # 累加评分数

# 转换为DataFrame并排序
genre_avg_ratings = {}
for genre, data in genre_ratings.items():
    genre_avg_ratings[genre] = {
        'avg_rating': round(np.mean(data['ratings']), 2),
        'votes': data['votes']  # 存储评分数而非作品数量
    }

genre_rating_df = pd.DataFrame.from_dict(genre_avg_ratings, orient='index')
genre_rating_df = genre_rating_df.sort_values('avg_rating', ascending=False).head(15)
visualization_data['genreRatingRelation'] = {
    'labels': genre_rating_df.index.tolist(),
    'data': genre_rating_df['avg_rating'].tolist(),
    'votes': genre_rating_df['votes'].tolist()  # 替换counts为votes
}

# 导演相关数据处理（同样修改为DOUBAN_VOTES）
director_ratings = {}
for idx, row in movies_df.iterrows():
    if pd.notna(row['DIRECTORS']) and pd.notna(row['DOUBAN_SCORE']) and pd.notna(row['DOUBAN_VOTES']):
        directors = row['DIRECTORS'].split('/')
        for director in directors:
            if director not in director_ratings:
                director_ratings[director] = {
                    'ratings': [],
                    'votes': 0
                }
            director_ratings[director]['ratings'].append(row['DOUBAN_SCORE'])
            director_ratings[director]['votes'] += int(row['DOUBAN_VOTES'])

director_avg_ratings = {}
for director, data in director_ratings.items():
    if len(data['ratings']) >= 3:
        director_avg_ratings[director] = {
            'avg_rating': round(np.mean(data['ratings']), 2),
            'votes': data['votes']
        }

director_rating_df = pd.DataFrame.from_dict(director_avg_ratings, orient='index')
director_rating_df = director_rating_df.sort_values('avg_rating', ascending=False).head(10)
visualization_data['directorRatingRelation'] = {
    'labels': director_rating_df.index.tolist(),
    'data': director_rating_df['avg_rating'].tolist(),
    'votes': director_rating_df['votes'].tolist()  # 替换counts为votes
}

# 演员与作品评分关系数据（前10个演员）
actor_ratings = {}
for idx, row in movies_df.iterrows():
    if pd.notna(row['ACTORS']) and pd.notna(row['DOUBAN_SCORE']) and pd.notna(row['DOUBAN_VOTES']):
        actors = row['ACTORS'].split('/')
        for actor in actors:
            if actor not in actor_ratings:
                actor_ratings[actor] = {
                    'ratings': [],
                    'votes': 0
                }
            actor_ratings[actor]['ratings'].append(row['DOUBAN_SCORE'])
            actor_ratings[actor]['votes'] += int(row['DOUBAN_VOTES'])

actor_avg_ratings = {}
for actor, data in actor_ratings.items():
    if len(data['ratings']) >= 3:
        actor_avg_ratings[actor] = {
            'avg_rating': round(np.mean(data['ratings']), 2),
            'votes': data['votes']
        }

actor_rating_df = pd.DataFrame.from_dict(actor_avg_ratings, orient='index')
actor_rating_df = actor_rating_df.sort_values('avg_rating', ascending=False).head(10)
visualization_data['actorRatingRelation'] = {
    'labels': actor_rating_df.index.tolist(),
    'data': actor_rating_df['avg_rating'].tolist(),
    'votes': actor_rating_df['votes'].tolist()
}

# 7. 演员-导演合作网络数据
cooperation_network = defaultdict(lambda: defaultdict(int))
for idx, row in movies_df.iterrows():
    if pd.notna(row['ACTORS']) and pd.notna(row['DIRECTORS']):
        actors = row['ACTORS'].split('/')
        directors = row['DIRECTORS'].split('/')
        for actor in actors:
            for director in directors:
                cooperation_network[actor][director] += 1

edges = []
nodes = set()
for actor, directors in cooperation_network.items():
    for director, count in directors.items():
        if count >= 2:  # 只保留合作2次以上的关系
            edges.append({
                'source': actor,
                'target': director,
                'value': count
            })
            nodes.add(actor)
            nodes.add(director)

nodes = [{'id': node, 'group': 1 if any(node in edge['source'] for edge in edges) else 2} for node in nodes]
visualization_data['cooperationNetwork'] = {
    'nodes': nodes,
    'links': edges
}


print("\n———————— 可视化数据 ————————")
for key, data in visualization_data.items():
    print(f"{key}: {data}")