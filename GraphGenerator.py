import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 

warnings.filterwarnings("ignore")

class GraphGenerator():
    def __init__(self):
        self.netflix_df = pd.read_csv('./databases/netflix_titles.csv')
        self.rt_movies = pd.read_csv('./databases/rotten_tomatoes_movies.csv')
        self.rt_movies = self.rt_movies.drop(columns=['rotten_tomatoes_link','critics_consensus','content_rating','genres','authors','actors','original_release_date','production_company','tomatometer_status','audience_status'])

    def __get_mean_count_rt_movies(self):
        df_clean = self.rt_movies.dropna(subset=['directors', 'tomatometer_rating', 'movie_title'])
        return df_clean.groupby('directors').agg(
            average_rating=('tomatometer_rating', 'mean'),
            movie_count=('tomatometer_rating', 'size')
        ).reset_index()

    def __clean_top_directors(self,df,top_number=20):
        df = df.sort_values(by='average_rating', ascending=False)
        df = df[df['movie_count'] > 3]
        return df.head(top_number)
    
    
    def __get_top_with_worst_best_movies(self,top_num):
        df = self.rt_movies.dropna()
        top_movies = df.loc[df.groupby('directors')['tomatometer_rating'].idxmax()].reset_index(drop=True)
        worst_movies = df.loc[df.groupby('directors')['tomatometer_rating'].idxmin()].reset_index(drop=True)

        top_movies = top_movies.dropna(subset=['directors', 'movie_title', 'tomatometer_rating'])
        worst_movies = worst_movies.dropna(subset=['directors', 'movie_title', 'tomatometer_rating'])
        df = self.__get_mean_count_rt_movies()
        df = self.__clean_top_directors(df,top_num)
        top_stats = df.merge(
            top_movies[['directors', 'movie_title', 'tomatometer_rating']],
            on='directors',
            suffixes=('', '_top_movie')
        ).rename(columns={
            'tomatometer_rating_top_movie': 'top_movie_rating',
            'movie_title': 'top_movie_title'
        })

        top_stats = top_stats.merge(
            worst_movies[['directors', 'movie_title', 'tomatometer_rating']],
            on='directors',
            suffixes=('', '_worst_movie')
        ).rename(columns={
            'tomatometer_rating_worst_movie': 'worst_movie_rating',
            'movie_title_worst_movie': 'worst_movie_title'
        })
        top_stats = top_stats.sort_values(by='average_rating', ascending=False)
        top_stats = top_stats.drop_duplicates()
        return top_stats

    def __truncate_text(self,text, max_length=12):
        if len(text) > max_length:
            return text[:max_length] + '...'
        return text



    def dist_num_rating(self):

        df = self.__get_mean_count_rt_movies()
        plt.figure(figsize=(12, 8))
        sns.regplot(data=df, x='movie_count', y='average_rating', scatter_kws={'s':100, 'edgecolor':'w'}, line_kws={'color':'red'})

        plt.title('Distribuição entre Nota Média x Qtd de filmes dirigidos')
        plt.xlabel('Quantidade de Filmes Dirigidos')
        plt.ylabel('Nota média')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def graph_best_worst(self,top_num):
        df = self.__get_top_with_worst_best_movies(top_num)
        plt.figure(figsize=(14, 8))
        plt.plot(df['directors'], df['tomatometer_rating'], marker='o', label='Melhor Filme Avaliado', color='blue')
        plt.plot(df['directors'], df['worst_movie_rating'], marker='o', label='Pior Filme Avaliado', color='red')
        for i in range(len(df)):
            plt.annotate(self.__truncate_text(df['top_movie_title'][i]), (df['directors'][i], df['tomatometer_rating'][i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=5)
            plt.annotate(self.__truncate_text(df['movie_title'][i]), (df['directors'][i], df['worst_movie_rating'][i]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=5)
        x_labels = [f"{name}\n({count} filmes, Média: {avg:.1f})" for name, count, avg in zip(df['directors'], df['movie_count'], df['average_rating'])]
        plt.xticks(ticks=range(len(df)), labels=x_labels, rotation=90, fontsize=10)
        plt.title('Filme melhor avaliado x Filme pior avaliado')
        plt.xlabel('Diretor')
        plt.ylabel('Nota')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def graph_count_rating(self):
        df = self.__get_mean_count_rt_movies()
        df = self.__clean_top_directors(df,20)
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='movie_count', y='average_rating', data=df, hue='directors', palette='tab20', s=100)
        for i in range(df.shape[0]):
            plt.annotate(f"{df['average_rating'].iloc[i]:.2f}",
                        (df['movie_count'].iloc[i], df['average_rating'].iloc[i]),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',fontsize=6)
        plt.title('Quantidade de Filmes x Nota Média')
        plt.xlabel('Quantidade de Filmes')
        plt.ylabel('Nota Média')
        plt.legend(title='Diretores', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

    def graph_critics_audience(self):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.histplot(self.rt_movies['tomatometer_rating'], kde=True, bins=20)
        plt.title('Distribuição das notas dos críticos')
        plt.xlabel('Nota dos críticos')
        plt.ylabel('Quantidade')
        plt.subplot(1, 2, 2)
        sns.histplot(self.rt_movies['audience_rating'], kde=True, bins=20)
        plt.title('Distribuição das notas da audiência')
        plt.xlabel('Nota da audiência')
        plt.ylabel('Quantidade')
        plt.show()
    
    def graph_movies_year(self):
        df = self.rt_movies
        df['streaming_release_date'] = pd.to_datetime(df['streaming_release_date'])
        df['year'] = df['streaming_release_date'].dt.year

        plt.figure(figsize=(12, 6))
        sns.countplot(x='year', data=df, palette='viridis')
        plt.title('Number of Movies Released by Year')
        plt.xlabel('Year')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()

    def top_movies_comparison(self):
        top_movies = self.rt_movies.nlargest(10, 'audience_count')
        top_movies = top_movies.drop(columns=['tomatometer_count','audience_count','tomatometer_top_critics_count','tomatometer_fresh_critics_count','tomatometer_rotten_critics_count'])
        top_movies.set_index('movie_title', inplace=True)
        top_movies.plot(kind='barh', figsize=(14, 8))
        plt.title('Nota dos Críticos x Nota da Audiência para o top10 filmes mais avaliados')
        plt.xlabel('Nota')
        plt.ylabel('Filme')
        plt.legend(title='Notas', labels=['Críticos','Audiência'], loc='upper left')
        plt.show()
    
    def graph_movie_time(self):
        df = self.rt_movies
        sns.histplot(df['runtime'].dropna(), bins=20, kde=True)
        plt.title('Distribuição de duração de filmes')
        plt.xlabel('Duração (em minutos)')
        plt.ylabel('Quantidade')
        plt.show()

    def netflix_releases(self):
        df = self.netflix_df[self.netflix_df['type'] == 'Movie']
        df['date_added'] = pd.to_datetime(df['date_added'])
        df['year_added'] = df['date_added'].dt.year

        sns.countplot(data=df, x='year_added', hue='type',legend=False)
        plt.title('Número de filmes adicionados a Netflix por Ano')
        plt.xlabel('Ano')
        plt.ylabel('Quantidade de Filmes')
        plt.show()

        plt.show()


    
    