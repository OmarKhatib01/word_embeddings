from preprocessing import parse_xml
import gensim

words = parse_xml()
# preprocessed_words = parse_xml()
# words = gensim.utils.simple_preprocess(preprocessed_words)
model = gensim.models.Word2Vec(words, min_count=1, workers=3, window=3, sg=1)

# model.train(words, total_examples=len(words), epochs=10)

# get vector for word
w0 = 'day'
print('vector for the word ' + w0 + ':',  model.wv[w0])


# gets 10 most similar words (10 is default)
w1 = 'woman'
print('10 most similar words to "woman" : ', model.wv.most_similar(positive=w1))

# gets top 6 similar words ('topn' parameter changes number of results)
w2 = 'day'
model.wv.most_similar(positive=w2, topn=6)


w3 = ['day', 'year']
print('10 most similar words to "day" OR "year" : ', model.wv.most_similar(positive=w3))





