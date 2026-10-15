# Retailrocket Session-kNN

Session-based k-Nearest-Neighbours recommender for the [Retailrocket
e-commerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset):
anonymous browsing events (`view` / `addtocart` / `transaction`) with no
persistent, reliable user identity across visits. Classical ML only -- no
neural networks, no gradient-descent-trained embeddings.

> Status: scaffolding in progress. See `paper/main.pdf` and the sections below
> once the pipeline lands.

## Why session-kNN

Retailrocket sessions are short, anonymous and highly recency-driven: a
`visitorid` is a cookie, not a durable account, and most sessions contain only
a handful of events before the visitor leaves. Collaborative filtering over a
persistent user-item matrix does not fit this regime. Session-kNN instead
represents the *current, partial* session as a (recency-weighted) set of
recently viewed items, retrieves the most similar historical sessions, and
lets those neighbours vote on the next item -- a strong, fully classical
baseline for session-based recommendation that needs no sequence model.

## Repository layout

```
conf/                 declarative feature specification (point-in-time)
data/                 .gitkeep only -- real/synthetic data is never committed
docs/                 architecture diagrams (drawio + rasterized svg/png)
paper/                LaTeX technical report, built reproducibly
scripts/              data generation, diagram export, table export
src/recommender/      sessionization, features, session-kNN, evaluation, serving
terraform/            Floci-first infrastructure (S3, DynamoDB, Lambda, EventBridge)
tests/                pytest suite incl. leakage/schema/parity tests
```

## License

MIT, see `LICENSE`.
