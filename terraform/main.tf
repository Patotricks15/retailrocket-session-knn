# Serving infrastructure for the session-kNN recommender.
#
# Design note: unlike a per-user matrix-factorization model, session-kNN has
# no stable per-entity key to precompute a top-N list against ahead of time
# -- a session does not exist until it starts, and a `visitorid` is an
# unreliable cookie. The only safe precompute target is therefore a single,
# session-independent popularity fallback used for empty/cold requests; every
# other request scores the posted partial session on-the-fly against the
# session-kNN index loaded from S3. That split is why this stack pairs a
# tiny DynamoDB table (one row) with a Lambda that does real work per
# request, rather than the "DynamoDB holds everything, Lambda is a GetItem"
# shape a per-user model would use.

locals {
  name_prefix = "${var.project_name}-${var.environment}"
}

# ---------------------------------------------------------------------------
# Artifacts bucket: the pickled session index, sessionized events, results
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "artifacts" {
  bucket        = "${local.name_prefix}-artifacts"
  force_destroy = true
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

// The floci/floci emulator does not converge the propagation check the AWS
// provider polls for here, so a local `apply` hangs for 3 minutes and then
// fails. The rule is real-AWS only; everything else in this stack runs in
// both modes.
resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  count  = var.use_floci ? 0 : 1
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "expire-old-model-versions"
    status = "Enabled"

    filter {
      prefix = "models/"
    }

    noncurrent_version_expiration {
      noncurrent_days = var.artifact_retention_days
    }
  }
}

# ---------------------------------------------------------------------------
# Fallback table: a single "GLOBAL" row holding the popularity top-N used
# when a request carries no recognised item (cold start / empty session)
# ---------------------------------------------------------------------------

resource "aws_dynamodb_table" "fallback" {
  name         = "${local.name_prefix}-fallback"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "session_key"

  attribute {
    name = "session_key"
    type = "S"
  }

  point_in_time_recovery {
    enabled = !var.use_floci
  }

  server_side_encryption {
    enabled = !var.use_floci
  }
}

# ---------------------------------------------------------------------------
# IAM for the inference Lambda
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "inference" {
  name               = "${local.name_prefix}-inference-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
}

data "aws_iam_policy_document" "inference" {
  # Read-only on the fallback table: the inference path never writes.
  statement {
    effect    = "Allow"
    actions   = ["dynamodb:GetItem"]
    resources = [aws_dynamodb_table.fallback.arn]
  }

  statement {
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.artifacts.arn}/models/*"]
  }

  statement {
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = ["arn:aws:logs:*:*:*"]
  }
}

resource "aws_iam_role_policy" "inference" {
  name   = "${local.name_prefix}-inference-policy"
  role   = aws_iam_role.inference.id
  policy = data.aws_iam_policy_document.inference.json
}

# ---------------------------------------------------------------------------
# Inference Lambda: on-the-fly session-kNN scoring
# ---------------------------------------------------------------------------

data "archive_file" "inference" {
  type        = "zip"
  source_file = "${path.module}/../src/recommender/serving/handler.py"
  output_path = "${path.module}/.build/inference.zip"
}

resource "aws_lambda_function" "inference" {
  function_name    = "${local.name_prefix}-inference"
  role             = aws_iam_role.inference.arn
  handler          = "handler.lambda_handler"
  runtime          = "python3.12"
  filename         = data.archive_file.inference.output_path
  source_code_hash = data.archive_file.inference.output_base64sha256
  timeout          = 10
  memory_size      = 256

  environment {
    variables = {
      FALLBACK_TABLE   = aws_dynamodb_table.fallback.name
      ARTIFACTS_BUCKET = aws_s3_bucket.artifacts.id
      MODEL_KEY        = "models/session_knn.pkl"
      DEFAULT_K        = "10"
    }
  }
}

resource "aws_cloudwatch_log_group" "inference" {
  name              = "/aws/lambda/${aws_lambda_function.inference.function_name}"
  retention_in_days = 14
}

# ---------------------------------------------------------------------------
# Nightly retraining trigger
#
# The handler above is request-time inference only; it does not retrain.
# EventBridge target is wired to the same function as a lightweight,
# infrastructure-level illustration of the nightly re-indexing cadence
# described in the paper. A production deployment would instead target a
# batch job (e.g. an ECS task or Glue job) that rebuilds the session index
# from fresh events and republishes `models/session_knn.pkl` to S3.
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_event_rule" "retrain" {
  name                = "${local.name_prefix}-retrain"
  description         = "Nightly session-kNN re-indexing trigger."
  schedule_expression = var.retraining_schedule
}

resource "aws_cloudwatch_event_target" "retrain" {
  rule = aws_cloudwatch_event_rule.retrain.name
  arn  = aws_lambda_function.inference.arn

  input = jsonencode({ action = "retrain" })
}

resource "aws_lambda_permission" "allow_events" {
  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.inference.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.retrain.arn
}
