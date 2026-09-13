# Semantic Book Map — 3D

An interactive 3D map of 100,000 books, positioned by semantic meaning rather than genre labels. Each point is a book; books with similar themes and content sit close together, purely based on an AI embedding model reading their descriptions.

Built as a way to explore whether embedding models can meaningfully organize a large, messy real-world dataset — and to validate that the result isn't just a pretty-looking random scatter.

## What it does

- Reads 100,000 English book descriptions from a Goodreads dataset
- Encodes each description into a 1024-dimensional embedding using `BAAI/bge-large-en-v1.5`
- Projects the embeddings into 3D space with UMAP for visualization
- Clusters books into 35 thematic groups with K-means
- Assigns each book a genre via zero-shot classification against a fixed genre list
- Precomputes the true top-8 most similar books for each title (exact cosine similarity, not the UMAP coordinates)
- Renders everything as an explorable 3D point cloud in the browser with Three.js — search, genre filtering, cluster legend, and a detail panel with the most similar books

## Live demo

Hosted via GitHub Pages: `https://<your-username>.github.io/<repo-name>/`

The page auto-loads `books_data_3d.json` from the same folder on load — no manual file upload needed. If that fetch fails (e.g. opened as a local file), it falls back to a drag-and-drop prompt.

## Repo structure

```
.
├── index.html            # 3D viewer (Three.js) — the whole frontend
├── books_data_3d.json    # exported data: coordinates, clusters, genres, similar books
├── books.ipynb           # full data pipeline (embeddings → UMAP → clustering → export)
└── README.md
```

## Running the pipeline yourself

1. Open `books.ipynb` and run the cells top to bottom.
2. Requires a CUDA GPU for reasonable embedding speed (100k descriptions takes a while on CPU). Tested on an RTX 4070 Super.
3. Dependencies:
   ```
   pip install datasets pandas numpy sentence-transformers umap-learn scikit-learn
   ```
4. The notebook saves intermediate results (`book_embeddings_bge.npy`, `book_sample.pkl`) so you don't have to recompute embeddings if you restart the session — see the "load already-saved embeddings" cell.
5. The final cell exports `books_data_3d.json`, which the frontend reads directly.

## Viewing the map locally

Because the frontend auto-fetches `books_data_3d.json` via `fetch()`, opening `index.html` directly as a local file (`file://`) will usually fail due to browser CORS restrictions on local fetches. To view it locally, serve the folder instead:

```
python3 -m http.server
```

Then open `http://localhost:8000`.

## Validating the results

Since "the map looks cool" isn't proof the embeddings are meaningful, the notebook includes an evaluation section:

- **Genre agreement@5**: 55.7% of a book's top-5 "most similar" neighbors share its genre, versus a **7.4% baseline** for random pairs — roughly a 7.5x lift over chance.
- **Top-1 similarity vs. random pairs**: nearest-neighbor similarity (0.75) is clearly separated from random-pair similarity (0.49), confirming the embedding space isn't degenerate.
- **Manual spot-checks**: random book samples show the model catching sub-genre-level structure (e.g. grouping werewolf-pack romance separately from general fantasy, or manga volumes together by format), not just surface genre.

## Tech stack

- **Embeddings**: `sentence-transformers` (`BAAI/bge-large-en-v1.5`)
- **Dimensionality reduction**: UMAP (`n_components=3`)
- **Clustering**: K-means (scikit-learn)
- **Frontend**: Three.js, vanilla JS, no build step

## Notes / known limitations

- The genre list is manually curated and may not perfectly cover every book (e.g. narrow nonfiction niches can get pushed into the "closest available" genre). See the notebook's genre-confidence check for details.
- Cluster count (K=35) was chosen as a practical balance between granularity and a readable color legend, not from a strong quantitative signal — silhouette scores for text embeddings are inherently low regardless of K.
