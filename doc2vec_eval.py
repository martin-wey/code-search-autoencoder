import argparse
import json
from pprint import pprint
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


def eval_doc2vec(model, train_set, test_set, ctx_size, k, output_path):
    sent_index = 0
    sent_count = 0
    next_word_accuracy = 0
    sum_reciprocal_rank = 0

    stop = False
    while not stop:
        full_sent = test_set[sent_index]
        if len(full_sent) >= ctx_size + 1:
            sent = full_sent[:ctx_size]
            # print('{} - Test sentence : {}'.format(sent_count, sent))
            inferred_vector = model.infer_vector(sent)
            sims_topn = model.docvecs.most_similar([inferred_vector], topn=k * 2)
            sims_topn_index = list(map(lambda x: x[0], sims_topn))

            suggestion = []
            for i, idx in enumerate(sims_topn_index):
                word_list = train_set[idx].words
                for word in word_list:
                    if word not in suggestion and len(suggestion) < k:
                        suggestion.append(word)
            next_word_in_suggestion = 1 if full_sent[ctx_size] in suggestion else 0
            if next_word_in_suggestion is 1:
                sum_reciprocal_rank += 1 / (suggestion.index(full_sent[ctx_size]) + 1)
            next_word_accuracy += next_word_in_suggestion
            sent_count += 1
        sent_index += 1
        if sent_index == (len(test_set) - 1):
            stop = True

    with open(output_path, 'a+') as f:
        f.write('Accuracy : {} (ctx={}, topn={}, test_size={})\n'.format(
            next_word_accuracy / sent_count,
            ctx_size,
            k,
            sent_count
        ))
        f.write('MRR : {} (ctx={}, topn={}, test_size={})\n'.format(
            sum_reciprocal_rank / sent_count,
            ctx_size,
            k,
            sent_count
        ))
    print('Accuracy : {} (ctx={}, topn={}, test_size={})\n'.format(
        next_word_accuracy / sent_count,
        ctx_size,
        k,
        sent_count
    ))
    print('MRR : {} (ctx={}, topn={}, test_size={})\n'.format(
        sum_reciprocal_rank / sent_count,
        ctx_size,
        k,
        sent_count
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', help='Path to the pre-trained doc2vec model.')
    parser.add_argument('-t', '--snippets_path', help='Path to the code snippets set (json format).')
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

    for idx in sims_topn_index:
        pprint(test_set[idx])
        print()
