Semantic Book Map — 3D

A 3D map of 100,000 books, where position is based on what each book is actually about — not genre tags. Books get placed close together when an embedding model reads their descriptions and finds them semantically similar.

What's in here
books.ipynb — the pipeline: loads book descriptions, embeds them with BAAI/bge-large-en-v1.5, reduces to 3D with UMAP, clusters with K-means, classifies genre (zero-shot), and finds each book's real nearest neighbors.
books_data_3d.json — the exported data the frontend reads.
index.html — the viewer, built with Three.js. Search, genre filter, cluster legend, click a point for details.
Demo

https://<your-username>.github.io/<repo-name>/

The page loads books_data_3d.json automatically as long as it's in the same folder. If it can't fetch it (e.g. you open the HTML file directly instead of through a server), it'll ask you to drop the file in manually.

Running the pipeline
pip install datasets pandas numpy sentence-transformers umap-learn scikit-learn

Run books.ipynb top to bottom. You'll want a GPU — encoding 100k descriptions on CPU is slow. Embeddings get saved to disk mid-way so you're not stuck re-running that step every time.

Viewing it locally

fetch() won't work if you just double-click index.html (browsers block that for local files). Serve it instead:

python3 -m http.server
Does the similarity actually mean anything?

Checked this before trusting it:

Books flagged as "most similar" share a genre 55.7% of the time, vs. 7.4% for random pairs.
Nearest-neighbor similarity (0.75) is clearly higher than random-pair similarity (0.49).
Manual spot-checks hold up too — e.g. it separates werewolf-pack romance from general fantasy, and groups manga by format even when the plots differ.
Known rough edges
The genre list is hand-picked and doesn't cover everything — some niche nonfiction gets pushed into the closest available genre.
K=35 clusters was picked for a readable legend, not because the math strongly favored it — silhouette scores are low across the board for text embeddings, so this came down to judgment.
