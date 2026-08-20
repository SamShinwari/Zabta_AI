# Zabta AI updated files

Replace these files in the repository:

- `src/fbr/citation.py`
- `src/fbr/retriever.py`
- `src/fbr/rate_resolver.py`

Then run:

```bash
bash zabta_updated/apply_test_updates.sh
```

The test-update script removes stale exact assertions for the old 187-PDF corpus and 153157-vector database. It does **not** change the production PDF discovery or vector-store behavior.

After applying all changes:

```bash
pytest tests/test_fbr_citation.py -v
pytest tests/test_rate_date_relevance.py -v
pytest tests/test_rate_retrieval.py -v
pytest tests/test_retrieval_quality.py -v -s
pytest tests/test_invoice_rate_resolver.py -v -s
pytest -v
```


## V2 retrieval ranking fix

For rate queries, retrieval now uses 65% semantic relevance and 35% document authority.
This prevents a high-semantic-score Tax Expenditure Report or other secondary document
from outranking the primary Sales Tax Act when their semantic scores are close.

Explicit "standard sales tax rate" queries are also forced through the rate-query ranking path.
