from sentence_transformers import SentenceTransformer
import numpy
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
sentences = []
f = open('NNTest.txt', encoding="utf8")
for line in f:
    sentences.append(line.strip())
print(len(sentences))
sentence_embeddings = model.encode(sentences)
output_file = "NNTest_bert.txt"
output = open(output_file,mode='w')
numpy.savetxt(output_file, sentence_embeddings)
print('save success')