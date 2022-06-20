# import libraries (you may add additional imports but you may not have to)
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


# get data files
!wget https://cdn.freecodecamp.org/project-data/books/book-crossings.zip

!unzip book-crossings.zip

books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'


# import csv data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})
    
    
    # add your code here - consider creating a new cell for each section of code

user_counts = df_ratings["user"].value_counts()
isbn_counts = df_ratings["isbn"].value_counts()

df_filter = df_ratings

df_filter = df_filter[~df_filter['user'].isin(user_counts[user_counts < 200].index)]
df_filter = df_filter[~df_filter['isbn'].isin(isbn_counts[isbn_counts < 100].index)]

df = pd.merge(left=df_books, right=df_filter, on="isbn")
df.drop_duplicates(subset=["title", "user"], inplace=True, keep="first")

pivot_table = pd.pivot_table(df, index="title", columns="user", values="rating", fill_value=0)
title_list = list(pivot_table.index.values)

print(pivot_table)
print(title_list)


neighbors = NearestNeighbors(n_neighbors=5, metric="cosine", algorithm="brute")
neighbors.fit(pivot_table.values)


# function to return recommended books - this will be tested

def get_index(title):
  return pivot_table.values[title_list.index(title)]


def get_recommends(book = ""):
  rating =  df[df["title"] == book].values[0][4]

  index = get_index("Where the Heart Is (Oprah's Book Club (Paperback))")
  distances, indices = neighbors.kneighbors([index])

  result = []
  recommended_books = [book, result]

  for distance, index in zip(distances[0], indices[0]):
    title = pivot.index.values[index]
    print(title)
    row = [title, distance]
    result.insert(0, row)

  return recommended_books

get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")


books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)

def test_book_recommendation():
  test_pass = True
  recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
  if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
    test_pass = False
  recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
  recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
  for i in range(2): 
    if recommends[1][i][0] not in recommended_books:
      test_pass = False
    if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
      test_pass = False
  if test_pass:
    print("You passed the challenge! 🎉🎉🎉🎉🎉")
  else:
    print("You haven't passed yet. Keep trying!")

test_book_recommendation()
