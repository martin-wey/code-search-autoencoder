import argparse
import json
from pprint import pprint
from gensim.models import Doc2Vec


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', help='Path to the pre-trained doc2vec model.')
    parser.add_argument('-t', '--snippets_path', help='Path to the code snippets set (json format).')
    parser.add_argument('-o', '--output_path', help='Results file path.')
    parser.add_argument('-q', '--query', help='Test query.')
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        print('Loading doc2vec model ...')
    model = Doc2Vec.load(args.model_path)
    if args.verbose:
        print('Loading code snippets ...')
    with open(args.snippets_path, 'r') as f:
        test_set = json.load(f)

    if args.verbose:
        print('Retrieving top k most similar code snippets ...')
    inferred_vector = model.infer_vector(args.query)
    sims_topn = model.docvecs.most_similar([inferred_vector], topn=10)
    sims_topn_index = list(map(lambda x: x[0], sims_topn))

    with open(args.output_path, 'a+') as fout:
        for idx in sims_topn_index:
            pprint(test_set[idx])
            fout.write(test_set[idx])
