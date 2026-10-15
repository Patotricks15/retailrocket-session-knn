"""Generate the two architecture diagrams as editable draw.io files.

drawpyo is already installed system-wide in this environment; if it is not,
install it into a dedicated venv so it never leaks into the project's own
dependencies:

    python -m venv .diagram-venv && .diagram-venv/bin/pip install drawpyo
    .diagram-venv/bin/python scripts/generate_diagrams.py

Then rasterize with the draw.io exporter:

    docker run --rm -v "$PWD/docs":/data -w /data rlespinasse/drawio-export \
      -f svg -o . --output-mode relative --remove-page-suffix .
    docker run --rm -v "$PWD/docs":/data -w /data rlespinasse/drawio-export \
      -f png -o . --output-mode relative --remove-page-suffix .

    Do not pass -t/--transparent to the PNG export: it produces a
    transparent background instead of the white one the README expects.

Finally, patch a literal white `<rect>` into both exported SVGs with
`scripts/patch_svg_background.py`, since GitHub's sanitizer strips the
inline `style="background:#fff"` that the exporter sets on the `<svg>` root.
"""

from __future__ import annotations

from pathlib import Path

import drawpyo
from drawpyo.diagram import Edge

DOCS = Path(__file__).resolve().parents[1] / "docs"

AWS_SHAPE = (
    "sketch=0;outlineConnect=0;fontColor=#232F3E;gradientColor=none;fillColor=#{fill};"
    "strokeColor=none;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;"
    "align=center;html=1;fontSize=11;fontStyle=0;aspect=fixed;"
    "shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.{icon};"
)
BOX = (
    "rounded=1;whiteSpace=wrap;html=1;fillColor=#{fill};strokeColor=#{stroke};"
    "fontSize=11;fontColor=#232F3E;align=center;verticalAlign=middle;"
)

ORANGE, BLUE, GREEN, PURPLE = "D45B07", "4D72F3", "277116", "8C4FFF"


def aws_node(page, title, icon, fill, x, y):
    node = drawpyo.diagram.Object(page=page, value=title)
    node.apply_style_string(AWS_SHAPE.format(fill=fill, icon=icon))
    node.position = (x, y)
    node.width, node.height = 78, 78
    return node


def box(page, title, x, y, width=190, height=58, fill="FFFFFF", stroke="9AA5B1"):
    node = drawpyo.diagram.Object(page=page, value=title)
    node.apply_style_string(BOX.format(fill=fill, stroke=stroke))
    node.position = (x, y)
    node.width, node.height = width, height
    return node


def link(page, source, target, label=""):
    """Connect two nodes.

    drawpyo's Edge has no apply_style_string, so styling is passed through the
    constructor instead.
    """
    return Edge(
        page=page,
        source=source,
        target=target,
        label=label,
        fontSize=10,
        fontColor="#4A5568",
        strokeColor="#4A5568",
        endArrow="blockThin",
        rounded=1,
    )


def build_aws_architecture() -> None:
    """Nightly re-indexing on the left, on-the-fly serving on the right."""
    file = drawpyo.File(file_name="aws-architecture.drawio", file_path=str(DOCS))
    page = drawpyo.Page(file=file, name="AWS architecture")

    box(page, "Offline — nightly re-indexing", 30, 20, 450, 36, fill="FFF4E6", stroke="D45B07")
    box(
        page, "Online — request-time scoring (no per-user precompute)",
        560, 20, 540, 36, fill="EEF2FF", stroke="4D72F3",
    )

    kaggle = box(page, "Retailrocket events\n(views · carts · purchases)", 30, 100, 210, 58)
    raw = aws_node(page, "S3\nraw", "bucket", GREEN, 310, 90)
    reindex = aws_node(page, "ECS Fargate\nsessionize + index", "machine_learning", ORANGE, 310, 260)
    schedule = aws_node(page, "EventBridge\nnightly 03:00", "event_time_based", PURPLE, 90, 260)
    artifacts = aws_node(page, "S3\nsession index (.pkl)", "bucket", GREEN, 310, 430)
    fallback = aws_node(page, "DynamoDB\npopularity fallback", "dynamodb", BLUE, 610, 430)

    client = aws_node(page, "Client\n(session so far)", "client", BLUE, 610, 90)
    api = aws_node(page, "API Gateway", "api_gateway", PURPLE, 830, 90)
    lam = aws_node(page, "Lambda\nsession-kNN scoring", "lambda_function", ORANGE, 830, 260)
    logs = aws_node(page, "CloudWatch\nlogs & metrics", "cloudwatch_2", PURPLE, 1020, 430)

    link(page, kaggle, raw, "ingest")
    link(page, schedule, reindex, "cron")
    link(page, raw, reindex, "read")
    link(page, reindex, artifacts, "session bags")
    link(page, reindex, fallback, "popularity top-N")
    link(page, client, api, "POST /recommend\n{items: [...]}")
    link(page, api, lam, "invoke")
    link(page, lam, artifacts, "load index\n(cold start)")
    link(page, lam, fallback, "GetItem\n(cold session)")
    link(page, lam, logs, "emit")

    file.write()
    print(f"wrote {DOCS / 'aws-architecture.drawio'}")


def build_recommendation_architecture() -> None:
    """The modelling pipeline, from raw events to a ranked next-item list.

    A single vertical spine with side branches, so that no edge has to cross
    a node box.
    """
    file = drawpyo.File(file_name="recommendation-architecture.drawio", file_path=str(DOCS))
    page = drawpyo.Page(file=file, name="Recommendation architecture")

    spine_x = 420
    w, gap = 260, 118

    raw = box(page, "Raw events\nvisitorid · timestamp · itemid · event", spine_x, 40, w, 72)
    sessionize = box(
        page,
        "Time-gap sessionization\ninactivity > 30 min → new session",
        spine_x, 40 + gap, w, 58, fill="FFF4E6", stroke="D45B07",
    )
    split = box(
        page,
        "Temporal session split\ntrain / val / test by session start_ts",
        spine_x, 40 + 2 * gap, w, 58, fill="FFF4E6", stroke="D45B07",
    )
    index = box(
        page,
        "Inverted item→session index\n(train sessions only)",
        spine_x, 40 + 3 * gap, w, 58, fill="E6FFFA", stroke="277116",
    )
    knn = box(
        page,
        "Session-kNN\nsim(S,N)=|S∩N|/√(|S||N|)\nw(N)=e^(−λΔt)",
        spine_x, 40 + 4 * gap, w, 86, fill="EEF2FF", stroke="4D72F3",
    )
    scoring = box(
        page,
        "Neighbour-vote scoring\ns(i,S)=Σ sim·w·1[i∈N]",
        spine_x, 55 + 5 * gap, w, 58, fill="EEF2FF", stroke="4D72F3",
    )
    evaluation = box(
        page,
        "Next-item evaluation\nRecall@K · MRR · Coverage\ncold-session (len=1) breakdown",
        spine_x, 70 + 6 * gap, w, 86, fill="FFF5F5", stroke="C53030",
    )

    features = box(
        page,
        "Point-in-time features\nconf/features.yaml\nsession length · elapsed time ·\nitem popularity (strict <)",
        spine_x - 330, 40 + 3 * gap - 6, 250, 86, fill="E6FFFA", stroke="277116",
    )
    baselines = box(
        page, "Baselines\nPopularity · Random",
        spine_x - 330, 70 + 6 * gap + 20, 250, 58, fill="F7FAFC",
    )
    serving = box(
        page, "Request-time Lambda\n+ popularity fallback (DynamoDB)",
        spine_x + 330, 40 + 4 * gap + 14, 250, 58, fill="EEF2FF", stroke="4D72F3",
    )

    link(page, raw, sessionize)
    link(page, sessionize, split)
    link(page, split, index, "train only")
    link(page, index, knn)
    link(page, knn, scoring)
    link(page, scoring, evaluation)
    link(page, split, features, "train only")
    link(page, features, knn, "context")
    link(page, baselines, evaluation)
    link(page, knn, serving, "on-the-fly")

    file.write()
    print(f"wrote {DOCS / 'recommendation-architecture.drawio'}")


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    build_aws_architecture()
    build_recommendation_architecture()
