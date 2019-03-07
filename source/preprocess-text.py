#!/usr/bin/python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for document classification, bitext filtering
# and mining
#
# --------------------------------------------------------
#
# Tool to calculate to embed a text file
# The functions can be also imported into another Python code


import re
import os
import tempfile
import sys
import time
import argparse
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn

# get environment
assert os.environ.get('LASER'), 'Please set the enviornment variable LASER'
LASER = os.environ['LASER']

sys.path.append(LASER + '/source/lib')
from text_processing import Token, BPEfastApply


SPACE_NORMALIZER = re.compile("\s+")
Batch = namedtuple('Batch', 'srcs tokens lengths')


def buffered_read(fp, buffer_size):
    buffer = []
    for src_str in fp:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LASER: Embed sentences')
    parser.add_argument('--input', type=str, required=True,
                        help='Input file to be encoded')
    parser.add_argument('--token-lang', type=str, default='--',
        help="Perform tokenization with given language ('--' for no tokenization)")
    parser.add_argument('--bpe-codes', type=str, default=None,
        help='Apply BPE using specified codes')
    parser.add_argument('-v', '--verbose', action='store_true',
        help='Detailed output')

    parser.add_argument('--buffer-size', type=int, default=10000,
        help='Buffer size (sentences)')
    parser.add_argument('--max-tokens', type=int, default=12000,
        help='Maximum number of tokens to process in a batch')
    parser.add_argument('--max-sentences', type=int, default=None,
        help='Maximum number of sentences to process in a batch')

    parser.add_argument('-o', '--output_dir', required=True,
        help='Output sentence embeddings')
    parser.add_argument('--cpu', action='store_true',
        help='Use CPU instead of GPU')
    args = parser.parse_args()

    args.buffer_size = max(args.buffer_size, 1)
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    ifname = args.input
    ifbasename = os.path.basename(args.input)
    if args.token_lang != '--':
        tok_fname = os.path.join(args.output_dir, ifbasename + '.tok')
        if not os.path.exists(tok_fname):

            Token(ifname,
                  tok_fname,
                  lang=args.token_lang,
                  romanize=True if args.token_lang == 'el' else False,
                  lower_case=True, gzip=False,
                  verbose=args.verbose, over_write=False)
        else:
            print(' - Tokenizer: {} exists. Skipping the tokenization and jump to BPEfying step.'.format(tok_fname))
        ifname = tok_fname

    if args.bpe_codes:
        bpe_fname = os.path.join(args.output_dir, ifbasename + '.bpe')
        if not os.path.exists(bpe_fname):
            BPEfastApply(ifname,
                         bpe_fname,
                         args.bpe_codes,
                         verbose=args.verbose, over_write=False)
        else:
            print(' - fast BPE: {} exists. Skipping the BPEfying step.'.format(tok_fname))

