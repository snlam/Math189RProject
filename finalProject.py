import gensim, numpy, matplotlib, pandas
import gzip, json
all_lines = []
for line in gzip.open("gutenberg-poetry-v001.ndjson.gz"):
     all_lines.append(json.loads(line.strip()))
