# The Math Library - Semantic search for competition math

Many years ago as high school students some friends and I had the idea to build a AI powered search engine specifically for mathematical competition problems. But just collecting a substantial database of problems was a major challenge and our text classification neural network could not reliable identify mathematical problems, so we abandoned the project after a few weeks. Now in light of the recent breakthroughs achieved by the transformer architecture in machine learning, I decided to revisit the idea.

The amazing folks at [Project Numina](https://projectnumina.ai/) have compiled a [database of around 900k math problems](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5), which means the tedious work of collecting high quality data was already completed. In the Jupyter notebook  the sentence transformer [nli-mpnet-base-v2](https://huggingface.co/sentence-transformers/nli-mpnet-base-v2) developed by [Nils Reimers and Iryna Gurevych](https://arxiv.org/pdf/1908.10084) is used to encode the semantic meaning of sentences and paragraphs into 768 dimensional vectors, which are saved into a [txtai](https://github.com/neuml/txtai) embedding database.

With `question(q)`, the user can enter their own problem or search query, which is then encoded into a high-dimensional vector. The semantically closest matches are then identified using the cosine-similarity test, which evaluates the database entries $v$ by:
```math
\cos(q, v) := \frac{\langle q, v \rangle}{\|q\| \cdot \|v\|} \in [-1, 1]
```

whereby a high score indicates that the respective vectors point in similar directions. The various examples with old problems from the "Bundeswettbewerb Mathematik" and the "Mathematik Olympiade" demonstrate that the search can either find the exact problem and its solution or at least identify very close matches, the solution of which can often be adapted to the original problem.

(c) Mia Müßig
