from flask import Flask
from flask import request
from flask import jsonify
# app = Flask(__name__)

from gensim import similarities
from gensim.models import ldamodel
import gensim.corpora as corpora
import pandas as pd
import json

from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

lda_model = ldamodel.LdaModel.load('ldamodel')
index = similarities.MatrixSimilarity.load('index')

# df = pd.read_json('https://raw.githubusercontent.com/cdap-39/data/master/data.json', orient='records')
df = pd.read_json('data.json', orient='records')

import os
os.environ.update({'MALLET_HOME': r'C:\\mallet-2.0.8\\mallet-2.0.8\\'})
mallet_path = 'C:\\mallet-2.0.8\\mallet-2.0.8\\bin\\mallet'


@app.route("/")
def hello():
    return "Hello World!"


@app.route('/similarity', methods=['POST'])  # GET requests will be blocked
def json_example():
    req_data = request.get_json()

    query = req_data['query']

    # query = "The Court of Appeal today (Jul 18) announced that the verdict on the case filed by the Attorney General against Ven. Galagodaaththe Gnanasara Thera for defaming the court will be announced on 8th of August.\r\n \r\nThis was ruled when the case was taken up before the President of the Court of Appeal, Justice Preethi Pathman Surasena and Justice Arjuna Obeysekere.\r\n \r\nOur correspondent said that Ven. Galagodaaththe Gnanasara Thera was present in the court when this order was given.Ven. Gnanasara Thera has been accused of contempt of court following his unruly behaviour on the day when journalist Prageeth Eknaligoda’s disappearance case was taken for hearing in Homagama Magistrate Court.Later, the Attorney General pressed charges Gnanasara Thera under four counts of charges including challenging the honour of the judiciary."

    id2word = corpora.Dictionary.load('id2')
    vec_bow = id2word.doc2bow(query.lower().split())

    vec_lda = lda_model[vec_bow]
    sims = index[vec_lda]

    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    top_ten = (sims[:5])

    print(top_ten[0][0])

    data = df.content.values.tolist()
    headings = df.heading.values.tolist()
    links = df.link.values.tolist()

    matches = []
    hit = {}

    i = 0
    while (i < len(top_ten)):
        record_index = top_ten[i][0]
        print(record_index)
        print(data[record_index])
        hit = {}
        hit['content'] = data[record_index]
        hit['heading'] = headings[record_index]
        hit['link'] = links[record_index]
        matches.append(hit)
        i += 1

    return jsonify(matches)


# if __name__ == 'main':
#     print('running...')
app.run()
