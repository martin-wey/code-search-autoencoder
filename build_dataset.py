import glob
import json
import ast
import argparse
from gensim.parsing.porter import PorterStemmer
import nlp_utils
from func_fetch_python import get_func_calls, get_func_def


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data_path', help='Path containing .jsonl dataset files.')
    parser.add_argument('-l', '--language', help='Language of the dataset (python or java)')
    parser.add_argument('-o', '--output_path', help='Path where to store extracted data.')
    parser.add_argument('-v', '--verbose', help='increase output verbosity', action="store_true")
    args = parser.parse_args()

    if args.verbose:
        print('Initialize tokenizers and porter stemmer ...')
    tokenizer_us = nlp_utils.initialize_nlp('en_core_web_sm')
    tokenizer_us = nlp_utils.initialize_tokenizer(tokenizer_us, 'underscore')
    tokenizer_camel = nlp_utils.initialize_nlp('en_core_web_sm')
    tokenizer_camel = nlp_utils.initialize_tokenizer(tokenizer_camel, 'camel')
    stemmer = PorterStemmer()

    func_sequences = []
    docstrings = []
    snippets = []
    if args.verbose:
        print('Extracting dataset ...')
    jsonl_files = glob.glob('{}/*.jsonl'.format(args.data_path))
    for jsonl_file in jsonl_files:
        if args.verbose:
            print('Parsing file : {}'.format(jsonl_file))
        with open(jsonl_file, 'r') as jsonl_data:
            data = list(jsonl_data)
        result = [json.loads(jline) for jline in data]

        if args.language == 'python':
            for func in result:
                func_data = []
                try:
                    tree = ast.parse(func['code'])
                except:
                    pass
                else:
                    func_doc_cleaned = nlp_utils.clean_text_data(stemmer, func['docstring_tokens'])
                    docstrings.append(func_doc_cleaned)
                    snippets.append(func['code'])

                    func_calls = get_func_calls(tree)
                    func_data.append(get_func_def(tree))
                    for fcall in func_calls:
                        elements = fcall.split('.')
                        func_data += elements
                    func_data = list(filter(lambda x: x is not None, func_data))

                    func_subtoken = nlp_utils.tokenize_list(tokenizer_us, func_data)
                    func_subtoken = nlp_utils.tokenize_list(tokenizer_camel, func_subtoken, lower=True)
                    func_subtoken = nlp_utils.strings_to_list(func_subtoken)
                    func_sequences.append(func_subtoken)

    if args.verbose:
        print('Exporting dataset ...')
    with open('{}/func_python.txt'.format(args.output_path), 'w+', encoding='utf-8') as fout:
        for item in func_sequences:
            fout.write(' '.join(item))
            fout.write('\n')

    with open('{}/docstrings_python.txt'.format(args.output_path), 'w+', encoding='utf-8') as fout:
        for item in docstrings:
            fout.write(' '.join(item))
            fout.write('\n')

    with open('{}/code_python.json'.format(args.output_path), 'w+', encoding='utf-8') as fout:
        json.dump(snippets, fout, indent=2)
    print('Done!')