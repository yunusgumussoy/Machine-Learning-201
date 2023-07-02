# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 23:45:59 2023

@author: Yunus

Lunapark Web3 Hub

Machine Learning 201
Week 1

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime
import dateutil.relativedelta as rd

# Numpy
# np.array(list)

l1 = [1, 2, 3]
np.array(l1)

# Array in range
# np.arrange(min, max, step)

np.arange(-12, 30, 5)

# Matrices
# np.ones((row, column)) np.zeros((row, column))

np.zeros((3, 255, 255))

# Random
# np.random.randint(low, high, size)

np.random.randint(0, 5, 3)

# Random Matrices
# np.random.randint(low, high, shape)

np.random.randint(10, 100, (5, 5))

# Generate Defined Number of Values in Range
# np.linspace(min, max, size)

# np.linspace() metodumuz iki sayı arasında verdiğimiz period sayısı kadar eşit aralıklarla sayı üretir.
# Yani linearly spaced array dediğimiz eşit aralıklı bir dizi oluşturur.
np.linspace(1, 50, 30)

# Reshape Arrays
# ndarray.reshape(shape)

m1 = np.random.randint(1, 100, (10, 10))
m1

# np.reshape() metodumuz ile dizimizi yeniden boyutlandırabiliriz.
# Örneğin 10x10'luk bir matrisimiz var ve bunu 25x4'lük bir matrise çevirmek istiyoruz.
# Bunun için reshape() metodunu kullanabiliriz.
m1.reshape((25,4))

# flatten() metodumuz ile matrisimizi tek boyutlu bir diziye çevirebiliriz.
m1.flatten()

# Indexing
# ndarray[row, column, depth]

m1[1, 9]

# Filtering and Masking
# ndarray[mask_condition]

m1

m1 > 50

# m1 matrisindeki 50'den büyük değerleri seçmek için mask oluşturduk.
mask1 = m1 > 50
m1[mask1]

# Statistical Operations
# ndarray.mean() ndarray.std() ndarray.var() ndarray.min() ndarray.max()

m1

display(m1.mean())

display(m1.std())

# m1 matrisindeki her bir satırın ortalaması 50'den büyük olan satırları seçmek için mask oluşturduk.
mask2 = [row.mean() > 50 for row in m1]
m1[mask2]

mask2

# Seed
# np.random.seed(seed)

np.random.seed(42)

# np.random.randint(low, high, size)

np.random.randint(0, 5, 5)

# Python'da tek satırda for döngüsü yazmak için list comprehension kullandık.
# Örneğin m1 matrisindeki her bir satırın toplamını bir liste içerisinde tutmak için şunu yazabiliriz:
[row.sum() for row in m1]

# Pandas
# pd.DataFrame(data, index, columns)

pd.Series(np.random.randint(0, 100, 100))

# Bir DataFrame'i içerisinde şu tür veriler ile oluşturabiliriz:
# - Numpy dizileri
# - Python sözlükleri
# - Pandas serileri
# - Pandas DataFrame'leri

# Örneğin bir DataFrame'i Numpy dizileri ile oluşturmak için şu şekilde bir kod yazabiliriz:
df = pd.DataFrame(np.random.randint(0, 100, (100, 5)))
df

# Columns
# DataFrame.columns

df.columns = ['A', 'B', 'C', 'D', 'E']
df

# Indexing
# DataFrame[column] DataFrame.loc[index] DataFrame.iloc[i_index]

df['B'][2]

df

# .loc indekslerin isimlerine göre seçim yapar
df.loc[1]

# .iloc indekslerin sırasına göre seçim yapar
df.iloc[-2]

# Operations

df['F'] = df['C'] + df['D']

df

df['F'].cumsum()

# DataFrame.itertuples()

# DataFrame üzeirnde iterasyon yapmamızı sağlayan metoddur. 
# Zorunda kalmadıkça yapmanızı önermediğimiz bir yöntemdir. 
# Eğer yapabiliyorsanız, vektörel işlemler kullanmanızı öneririz. 
for row in df.itertuples():
    print(row.A)

# Adding and Removing columns
# DataFrame.drop()

df

# Burada önemli bir parametre olan axis parametresini görüyoruz.
# Bu parametre ile satırlar üzerinde veya sütunlar üzerinde işlem yapabiliriz.
# axis=0 satırlar üzerinde işlem yapar
# axis=1 sütunlar üzerinde işlem yapar

# Örneğin burada sütunların üzerinde işlem yapıyoruz ve D sütununu siliyoruz.
df.drop('D', axis=1)

# Describe
# DataFrame.describe() DataFrame.info() DataFrame.value_counts() DataFrame.nunique()

# pd.notna(Dataframe)

dict1 = {'column1': ['A', 'A', np.nan, 'C'],
         'column2': [2, 2.3, 5.8, -2],
         'column3': ['Yunus', 'Çağan', 'Beril', 'Alper']}

# Mesela bir DataFrame'i Python sözlükleri ile oluşturmak için şu şekilde bir kod yazabiliriz:
df2 = pd.DataFrame(dict1)
df2

# DataFrame.info() metodu ile DataFrame'imizin içerisindeki verilerin türlerini, eksik değerlerin olup olmadığını,
# ve DataFrame'in boyutlarını görebiliriz.
df2.info()

df.mean() > 50

# Burada yine karşımıza önemli bir opraatör çıkıyor. (:) operatörü.
# Bu operatör ile DataFrame'in içerisindeki verileri seçebiliriz.
# Örneğin burada df DataFrame'inin içerisindeki tüm satırları seçiyoruz.
# Çünkü satırlar için hiçbir şey belirtmedik ve hepsini işleme dahil etmek istiyoruz.
# Sütunlar için ise df.mean() > 50 ifadesi ile sütunların ortalamalarını hesapladık ve 50'den büyük olanları seçtik.

df.loc[:, df.mean() > 50].max(axis=1)

# Burada (:) operatörü ile yine tüm satırları seçiyoruz.
# fakat sütunların yalnızca bir kısmını seçiyoruz.
np.random.randint(0, 5, (5, 5))[:, 3:4]

# Write and Read
# DataFrame.to_csv(file_path) pd.read_csv(file_path)

df

# DataFrame'in içerisindeki verileri CSV dosyasına yazmak için şu şekilde bir kod yazabiliriz:
df.to_csv('deneme.csv', index=False)

# CSV dosyasındaki verileri DataFrame olarak okumak için şu şekilde bir kod yazabiliriz:
pd.read_csv('deneme.csv')

# DateTime
# pd.date_range(start, end, frequency, period)

# pandas.date_range() metodu ile belli bir aralıkta tarihler oluşturabiliriz.
# Ve bunu frequency ve periods parametreleri ile istediğimiz sıklıkta yapabiliriz.
pd.date_range('2020-03-01', '2020-03-31', periods=5)

# pandas.to_datetime() metodu ile string tarihleri Timestamp objelerine çevirebiliriz.
pd.to_datetime('2023-03-30')

# Burada datetime ve relativedelta modüllerini kullanarak bir ay önceki tarihi bulduk.
datetime.datetime.now() - rd.relativedelta(months=1)

# MatPlotLib
# Basic Plots
# plt.plot() plt.scatter() plt.bar() plt.stem() plt.step() plt.fill_between()

plt.plot(np.random.randint(0, 10, 90))
# plt.plot(np.random.randint(0, 10, 90))

plt.show()

plt.figure(figsize=(5, 5))

x = np.linspace(0, 100, 1000)

y1 = [2 * np.math.cos(2*a) for a in x]
y2 = [2 * np.math.sin(2*a) for a in x]

plt.plot(y1, y2)
plt.show()

plt.scatter(x, y1)
plt.show()

plt.bar(x=[1, 2, 3, 4, 5], height=[50, 60, 80, 50, 72])
plt.show()

df

# Gördüğünüz gibi DataFrame'in içerisindeki verileri çizdirmek için dataframe objesinin kendi metodu da var.
df.plot(kind='scatter', x='A', y='B')

# Statistical Plots
# plt.hist() plt.pie() plt.boxplot()

plt.hist(y1, bins=100)
plt.show()








