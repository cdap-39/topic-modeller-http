from gensim import similarities
from gensim.models import ldamodel
import gensim.corpora as corpora
import pandas as pd

lda_model = ldamodel.LdaModel.load('ldamodel')
index = similarities.MatrixSimilarity.load('index')

df = pd.read_json(
    'https://raw.githubusercontent.com/cdap-39/data/master/newsfirst_hirunews.json')

import os
os.environ.update({'MALLET_HOME': r'C:\\mallet-2.0.8\\mallet-2.0.8\\'})
mallet_path = 'C:\\mallet-2.0.8\\mallet-2.0.8\\bin\\mallet'

query = "The Joint Postal Trade Union Front says that they would call on the Chief Prelates of the Asgiri and Malwathu chapters to make aware of their ongoing struggle. The Postal employees are continuing their strike for the 14th day over several demands including a service structure which is unique for the Postal Department."

id2word = corpora.Dictionary.load('id2')
vec_bow = id2word.doc2bow(query.lower().split())

vec_lda = lda_model[vec_bow]
sims = index[vec_lda]

sims = sorted(enumerate(sims), key=lambda item: -item[1])
top_ten = (sims[:5])

print(top_ten[0][0])

print('\n################################################################################')
i = 0
while (i < len(top_ten)):
    record_index = top_ten[i][0]
    print('\n' + df['data'][record_index])
    i += 1

# print(df['data'][10])
# print(df['data'][19])
# print(df['data'][7])
# print(df['data'][13])
# print(df['data'][8])
# print(df['data'][33])
# print(df['data'][3])
# print(df['data'][24])
# print(df['data'][108])
# print(df['data'][49])
