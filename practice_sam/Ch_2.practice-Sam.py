#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 09:49:40 2023
핸즈온 머신러닝 3판 2
@author: sl
"""

#%%
#Scikit Learn 버전 확인해보기 

import pandas as pd 
from pathlib import Path
import tarfile
import urllib.request

#%%
# 데이터 불러오기 
def load_housing_data():
    file_Path = Path('datasets/housing.tgz')#압축파일이 있는지 확인
    if not file_Path.is_file():
        Path('datasets').mkdir(parents=True, exist_ok= True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, file_Path)
        with tarfile.open(file_Path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
            print("Successfully creat file!!!")
    else :
        print("File is already exist in ",file_Path, "\n  Use original file !!")
        
    return pd.read_csv(Path('datasets/housing/housing.csv'))
    
#%%    
housing =  load_housing_data()

#%%
# No.2 데이터 구조 훓어 보기 

housing.head(5)
#%%

housing.info()

#%%
# 특정 컬럼값의 반복 확인 하기 
housing['ocean_proximity'].value_counts()

#%%
#전체 숫자형 데이터 들의 통계 요약
#통계 Null값은 제외 
describ_data = housing.describe()

#%%
#그래프로 표시하기 
import matplotlib.pyplot as plt 
#%%
housing.hist(bins=50, figsize=(12,8))
plt.show()



#%%
#데이터 세트 만들기 
#임시 랜덤선택 하기 
import numpy as np 
#%%

shuffled_indices = np.random.permutation(len(housing))

print(shuffled_indices)

#%%

def shuffle_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[test_indices], data.iloc[train_indices]
    
#%%
#Pandas 행,열 선택해보기 예제 - 실험용 
test = [[1,2,3],[4,5,6],[7,8,9]]

df = pd.DataFrame(test)

print (df)

print("="*20)
print(df.iloc[0])

#%%
test_samples , train_samples = shuffle_split_data(housing, 0.2)

print (train_samples.shape)
print(test_samples.shape)

#%%
#Scikit Learn 에 내장함수를 사용하기 
#모듈명 : model_selection
#함수명 : train_test_split   : Split arrays or matrices into random train and test subsets.

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing,test_size=0.2, random_state=42)
#%%
print("train_set :",train_set.shape,"test_set :",test_set.shape)

#%%
#샘플링의 편향성 제거 하기 
#소득"median_income" 을 5단계로 나눠서 Categorize 시키자 
# pd.cut () , data, [단게], [단계별레이블(이름)]

#새로운 Column 'income_cat' 를 만들어 값을 대입해준다. 

housing['income_cat'] = pd.cut(housing["median_income"], bins=[0.0, 1.5, 3.0, 4.5, 6, np.inf], labels=[1,2,3,4,5])

#%%

housing['income_cat'].value_counts()

#%%
# income_cat 별 실제 가격 분표 상황 파악하기 
housing[housing['income_cat'] == 5]['median_income'].value_counts(())


#%%
housing['income_cat'].value_counts().plot.bar()

 #%%
#그래프 index 로 정렬하기 
 
housing['income_cat'].value_counts().sort_index().plot.bar()
#%%
# grid = True 
housing['income_cat'].value_counts().sort_index().plot.bar(rot=0, grid= True)
plt.xlabel("middle income catagory")
plt.ylabel("amount")
plt.show()
#%%
#각각 다른 10개의 셋트로 분활 하기 (참조)

from sklearn.model_selection import train_test_split

strat_train_set , strat_test_set = train_test_split(housing, test_size=0.2,random_state=42,stratify=housing['income_cat'])

#%%
strat_train_set['income_cat'].value_counts() / len(strat_test_set)


#%%
#modelSelectin 에서 각종 그룹 Spliter 함수 알아보기 

# No.1 GroupShuffleSplit


from sklearn.model_selection import GroupShuffleSplit

X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
group = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
#%%
gss = GroupShuffleSplit(n_splits=4, test_size=0.5, random_state=24)
for train, test in gss.split(X,y,groups= group):
    print(train,":",test)
#%%
# Train 데이터셋은 원본에서 복사하여 사용하자 
housing2 = strat_train_set.copy()

#%%
#지리적 데이터 시각화 하기 
#위도 경도 데이터 사용 , plot (산점도 사용)

housing2.plot(kind='scatter',)






