from gensim.parsing.porter import PorterStemmer
from nlp_utils import clean_text_data

with open('./resources/queries.txt', 'r') as fin:
    data = fin.readlines()

stemmer = PorterStemmer()
queries = []
for query in data:
    query_list = query.split()
    query_cleaned = clean_text_data(stemmer, query_list)

    for term in query_cleaned:
        if term not in query_list:
            query_list.append(term)

    queries.append(query_list)

with open('./resources/queries_cleaned.txt', 'w+') as fout:
    for item in queries:
        fout.write(' '.join(item))
        fout.write('\n')