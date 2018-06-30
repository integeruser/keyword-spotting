#!/usr/bin/env python3
import argparse
import json

import construct_corpus
import construct_codebook
import find_matches
import extract_matches


parser = argparse.ArgumentParser(description="")
parser.add_argument("-o", default="", help="")
parser.add_argument("-f", action="store_true", default=False, help="")
parser.add_argument("config_file_path", help="")
args = parser.parse_args()

# load config file
with open(args.config_file_path) as f:
    config = json.load(f)

# run scripts
corpus_file_path = construct_corpus.run(config, output_directory_path=args.o, force=args.f)
config["corpus_file_path"] = corpus_file_path

codebook_file_path = construct_codebook.run(config, output_directory_path=args.o, force=args.f)
config["codebook_file_path"] = codebook_file_path

matches_file_path = find_matches.run(config, output_directory_path=args.o, force=args.f)
config["matches_file_path"] = matches_file_path

extract_matches.run(config, output_directory_path=args.o, force=args.f)
