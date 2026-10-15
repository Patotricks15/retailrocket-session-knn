# Infrastructure

Terraform for the serving path: an S3 bucket for the sessionized events and
the pickled session-kNN index, a single-row DynamoDB "popularity fallback"
table, and a Lambda behind IAM that scores partial sessions on request.

The default profile targets the local **Floci** AWS emulator, so `apply`
costs nothing and needs no AWS account.

## Local (default)

```bash
docker compose up -d                       # start Floci on :4566
terraform init
terraform validate
terraform apply -auto-approve
terraform output
terraform destroy -auto-approve
docker compose down
```

## Real AWS

The same configuration deploys to a real account by flipping one variable:

```bash
terraform apply -var use_floci=false -var aws_region=sa-east-1
```

With `use_floci = false` the provider stops injecting dummy credentials
and custom endpoints, and the two features the Floci emulator does not
implement (DynamoDB point-in-time recovery and server-side encryption)
switch on.

## Variables

| Variable | Default | Purpose |
|---|---|---|
| `project_name` | `retailrocket-knn` | Prefix for every resource name |
| `environment` | `dev` | Environment tag and name suffix |
| `aws_region` | `us-east-1` | Target region |
| `use_floci` | `true` | Target the local Floci emulator instead of real AWS |
| `floci_endpoint` | `http://localhost:4566` | Floci emulator address |
| `artifact_retention_days` | `90` | Expiry for non-current model versions in S3 |
| `retraining_schedule` | `cron(0 3 * * ? *)` | Nightly re-indexing trigger |

## Design notes

**Why score in Lambda instead of precomputing a top-N table?** Session-kNN
has no stable per-entity key to precompute against: a session does not exist
until it starts, and `visitorid` is an unreliable cookie, not a durable
account. The only recommendation surface that is safe to precompute is a
single, session-independent popularity list for empty/cold requests
(`fallback` table, one row, key `"GLOBAL"`). Every other request posts the
items viewed so far and is scored on-the-fly against the session index
pulled from S3 at cold start -- the architecturally honest shape for an
anonymous, request-time method, unlike the BPR/SVD portfolio projects where a
stable `user_id` lets the ranking be fully precomputed offline.

**Why `PAY_PER_REQUEST`?** Portfolio traffic is bursty and near zero on
average. On-demand billing avoids paying for provisioned capacity that sits
idle, and removes capacity planning from the deployment.

**IAM scope.** The inference role is read-only: `GetItem` on the fallback
table and `GetObject` restricted to the `models/` prefix. The request path
never writes, so it is not granted write permissions.

## Cost

Everything above is free on Floci. On real AWS at portfolio traffic:
S3 under $0.05/month for less than 1 GB, DynamoDB on-demand under $1/month,
and Lambda within the free tier. `terraform destroy` plus `docker compose
down` removes all of it.
