from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Any, Set, Tuple, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Configuration
DB_PATH = os.environ.get("LEADERBOARD_DB", "leaderboard.db")
GROUND_TRUTH_PATH = os.environ.get("GROUND_TRUTH_PATH", "ground_truth.txt")
NUM_DATASETS = 4

# Dataset sizes (N_1, N_2, N_3, N_4) - can be overridden via environment variables
DATASET_SIZES = [
    int(os.environ.get("DATASET1_SIZE", "400")),
    int(os.environ.get("DATASET2_SIZE", "400")),
    int(os.environ.get("DATASET3_SIZE", "400")),
    int(os.environ.get("DATASET4_SIZE", "400")),
]

# IMPORTANT: set this to your GitHub Pages origin
# For local testing, also include http://localhost:8080
ALLOWED_ORIGINS = [
    "https://nerdslab.github.io",
    "http://localhost:8080",  # Local development
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, data: Dict[str, Any]):
        if self.active_connections:
            # Create a list of tasks to send to all connections
            disconnected = set()
            for connection in self.active_connections:
                try:
                    await connection.send_json(data)
                except Exception:
                    disconnected.add(connection)
            # Remove disconnected connections
            for conn in disconnected:
                self.active_connections.discard(conn)

manager = ConnectionManager()

def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """
    Initialize database schema. If old schema exists, it will be migrated.
    """
    with db() as conn:
        # Check if table exists and what columns it has
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='submissions'"
        )
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            # Check if old schema (has 'score' column instead of 'score_dataset1')
            cursor = conn.execute("PRAGMA table_info(submissions)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'score' in columns and 'score_dataset1' not in columns:
                # Old schema detected - drop and recreate
                print("Old schema detected. Migrating database...")
                conn.execute("DROP TABLE submissions")
                table_exists = False
        
        if not table_exists:
            conn.execute(
                """
                CREATE TABLE submissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    score_dataset1 REAL NOT NULL,
                    score_dataset2 REAL NOT NULL,
                    score_dataset3 REAL NOT NULL,
                    score_dataset4 REAL NOT NULL,
                    avg_rank REAL NOT NULL,
                    submitted_at TEXT NOT NULL,
                    file_sha256 TEXT NOT NULL
                )
                """
            )
            # Create index for faster rank computation
            conn.execute(
                "CREATE INDEX idx_avg_rank ON submissions(avg_rank)"
            )
        else:
            # Table exists with new schema - just ensure index exists
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_avg_rank ON submissions(avg_rank)"
            )
        conn.commit()

init_db()

# Ground truth labels (loaded once at startup)
_ground_truth: Optional[List[List[str]]] = None

def load_ground_truth() -> List[List[str]]:
    """Load ground truth labels from file. Creates synthetic file if missing."""
    global _ground_truth
    if _ground_truth is not None:
        return _ground_truth
    
    if not os.path.exists(GROUND_TRUTH_PATH):
        # Create synthetic ground truth
        print(f"Creating synthetic ground truth file: {GROUND_TRUTH_PATH}")
        labels = []
        for d_idx, size in enumerate(DATASET_SIZES):
            # Generate synthetic labels (e.g., "class_0", "class_1", etc.)
            for i in range(size):
                labels.append(f"class_{i % 3}")  # 3 classes per dataset
        
        with open(GROUND_TRUTH_PATH, 'w') as f:
            f.write('\n'.join(labels))
    
    # Load ground truth
    with open(GROUND_TRUTH_PATH, 'r') as f:
        all_labels = [line.strip() for line in f if line.strip()]
    
    # Split into 4 datasets
    _ground_truth = []
    start = 0
    for size in DATASET_SIZES:
        _ground_truth.append(all_labels[start:start + size])
        start += size
    
    return _ground_truth

def parse_labels_file(content: bytes) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Parse labels file (one label per line) and split into 4 datasets.
    Returns: (labels1, labels2, labels3, labels4)
    """
    try:
        text = content.decode('utf-8').strip()
        all_labels = [line.strip() for line in text.split('\n') if line.strip()]
    except UnicodeDecodeError:
        raise ValueError("File must be valid UTF-8 encoded text")
    
    expected_total = sum(DATASET_SIZES)
    if len(all_labels) != expected_total:
        raise ValueError(
            f"Expected {expected_total} labels (sum of dataset sizes), got {len(all_labels)}"
        )
    
    # Split into 4 datasets
    start = 0
    labels1 = all_labels[start:start + DATASET_SIZES[0]]
    start += DATASET_SIZES[0]
    labels2 = all_labels[start:start + DATASET_SIZES[1]]
    start += DATASET_SIZES[1]
    labels3 = all_labels[start:start + DATASET_SIZES[2]]
    start += DATASET_SIZES[2]
    labels4 = all_labels[start:]
    
    return labels1, labels2, labels3, labels4

def compute_accuracy(submitted: List[str], ground_truth: List[str]) -> float:
    """Compute accuracy score (fraction of correct predictions)."""
    if len(submitted) != len(ground_truth):
        raise ValueError(f"Label count mismatch: {len(submitted)} vs {len(ground_truth)}")
    correct = sum(1 for s, g in zip(submitted, ground_truth) if s == g)
    return correct / len(ground_truth) if ground_truth else 0.0

def compute_scores(submitted_labels: Tuple[List[str], List[str], List[str], List[str]]) -> Tuple[float, float, float, float]:
    """
    Compute accuracy scores for all 4 datasets.
    Returns: (score1, score2, score3, score4)
    """
    gt = load_ground_truth()
    return (
        compute_accuracy(submitted_labels[0], gt[0]),
        compute_accuracy(submitted_labels[1], gt[1]),
        compute_accuracy(submitted_labels[2], gt[2]),
        compute_accuracy(submitted_labels[3], gt[3]),
    )

def recompute_ranks() -> None:
    """
    Recompute average ranks for all submissions and update the database.
    Rank is computed as: for each dataset, rank submissions by score (desc),
    then average the 4 ranks for each submission.
    Lower avg_rank is better (rank 1 is best).
    """
    with db() as conn:
        # Get all submissions with their scores
        rows = conn.execute(
            """
            SELECT id, score_dataset1, score_dataset2, score_dataset3, score_dataset4
            FROM submissions
            """
        ).fetchall()
        
        if not rows:
            return
        
        # Build lists of scores for each dataset
        dataset_scores = [[] for _ in range(NUM_DATASETS)]
        submission_ids = []
        
        for row in rows:
            submission_ids.append(row["id"])
            dataset_scores[0].append((row["id"], row["score_dataset1"]))
            dataset_scores[1].append((row["id"], row["score_dataset2"]))
            dataset_scores[2].append((row["id"], row["score_dataset3"]))
            dataset_scores[3].append((row["id"], row["score_dataset4"]))
        
        # Compute ranks for each dataset (higher score = better = lower rank)
        # Handle ties by giving same rank (standard competition ranking)
        dataset_ranks = [{} for _ in range(NUM_DATASETS)]
        
        for d_idx in range(NUM_DATASETS):
            # Sort by score descending
            sorted_scores = sorted(dataset_scores[d_idx], key=lambda x: x[1], reverse=True)
            
            current_rank = 1
            for i, (sub_id, score) in enumerate(sorted_scores):
                if i > 0 and score < sorted_scores[i-1][1]:
                    # Score decreased, so rank increases
                    current_rank = i + 1
                dataset_ranks[d_idx][sub_id] = current_rank
        
        # Compute average rank for each submission
        avg_ranks = {}
        for sub_id in submission_ids:
            ranks = [dataset_ranks[d][sub_id] for d in range(NUM_DATASETS)]
            avg_ranks[sub_id] = sum(ranks) / NUM_DATASETS
        
        # Update database
        for sub_id, avg_rank in avg_ranks.items():
            conn.execute(
                "UPDATE submissions SET avg_rank = ? WHERE id = ?",
                (avg_rank, sub_id)
            )
        conn.commit()

async def broadcast_leaderboard_update():
    """
    Recompute ranks, fetch leaderboard, and broadcast to all WebSocket clients.
    This is called after each submission, with lock protection to avoid race conditions.
    """
    async with manager.lock:
        # Recompute all ranks
        recompute_ranks()
        
        # Fetch updated leaderboard
        with db() as conn:
            rows = conn.execute(
                """
                SELECT name, score_dataset1, score_dataset2, score_dataset3, score_dataset4, avg_rank, submitted_at
                FROM submissions
                ORDER BY avg_rank ASC, submitted_at DESC
                LIMIT 200
                """
            ).fetchall()
        
        entries = [
            {
                "name": r["name"],
                "score_dataset1": float(r["score_dataset1"]),
                "score_dataset2": float(r["score_dataset2"]),
                "score_dataset3": float(r["score_dataset3"]),
                "score_dataset4": float(r["score_dataset4"]),
                "avg_rank": float(r["avg_rank"]),
                "submitted_at": r["submitted_at"],
            }
            for r in rows
        ]
        
        # Broadcast to all connected clients
        await manager.broadcast({"entries": entries})

@app.post("/api/submit")
async def submit(name: str = Form(...), file: UploadFile = File(...)) -> Dict[str, Any]:
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")

    # Basic upload guardrails
    content = await file.read()
    if len(content) > 5_000_000:
        raise HTTPException(status_code=413, detail="File too large (max 5MB).")

    # Parse labels file
    try:
        labels = parse_labels_file(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Compute scores for all 4 datasets
    score1, score2, score3, score4 = compute_scores(labels)
    
    file_sha = hashlib.sha256(content).hexdigest()
    submitted_at = datetime.now(timezone.utc).isoformat()
    
    # Insert submission (avg_rank will be computed later)
    with db() as conn:
        cur = conn.execute(
            """
            INSERT INTO submissions 
            (name, score_dataset1, score_dataset2, score_dataset3, score_dataset4, avg_rank, submitted_at, file_sha256)
            VALUES (?, ?, ?, ?, ?, 999.0, ?, ?)
            """,
            (name, float(score1), float(score2), float(score3), float(score4), submitted_at, file_sha),
        )
        submission_id = cur.lastrowid
    
    # Trigger rank recomputation and broadcast (with lock protection)
    await broadcast_leaderboard_update()
    
    # Fetch the computed avg_rank
    with db() as conn:
        row = conn.execute(
            "SELECT avg_rank FROM submissions WHERE id = ?",
            (submission_id,)
        ).fetchone()
        avg_rank = float(row["avg_rank"]) if row else None
    
    return {
        "submission_id": submission_id,
        "name": name,
        "score_dataset1": score1,
        "score_dataset2": score2,
        "score_dataset3": score3,
        "score_dataset4": score4,
        "avg_rank": avg_rank,
        "submitted_at": submitted_at,
    }

@app.get("/api/leaderboard")
def leaderboard() -> Dict[str, List[Dict[str, Any]]]:
    """
    Get current leaderboard sorted by avg_rank (ascending, lower is better).
    """
    with db() as conn:
        rows = conn.execute(
            """
            SELECT name, score_dataset1, score_dataset2, score_dataset3, score_dataset4, avg_rank, submitted_at
            FROM submissions
            ORDER BY avg_rank ASC, submitted_at DESC
            LIMIT 200
            """
        ).fetchall()

    entries = [
        {
            "name": r["name"],
            "score_dataset1": float(r["score_dataset1"]),
            "score_dataset2": float(r["score_dataset2"]),
            "score_dataset3": float(r["score_dataset3"]),
            "score_dataset4": float(r["score_dataset4"]),
            "avg_rank": float(r["avg_rank"]),
            "submitted_at": r["submitted_at"],
        }
        for r in rows
    ]
    return {"entries": entries}

@app.websocket("/ws/leaderboard")
async def websocket_leaderboard(websocket: WebSocket):
    """
    WebSocket endpoint for real-time leaderboard updates.
    Clients connect here and receive updates whenever a new submission is made.
    """
    # Check origin for WebSocket connections (more lenient for localhost)
    origin = websocket.headers.get("origin") or websocket.headers.get("Origin") or ""
    
    # Allow localhost with any port for local development, or allowed origins
    is_localhost = origin.startswith("http://localhost:") or origin.startswith("http://127.0.0.1:")
    is_allowed = origin in ALLOWED_ORIGINS or is_localhost or not origin  # Allow if no origin header (some clients don't send it)
    
    if not is_allowed:
        print(f"WebSocket rejected: origin={origin}")
        await websocket.close(code=1008, reason="Origin not allowed")
        return
    
    # Accept the connection
    await manager.connect(websocket)
    try:
        # Send initial leaderboard on connect
        with db() as conn:
            rows = conn.execute(
                """
                SELECT name, score_dataset1, score_dataset2, score_dataset3, score_dataset4, avg_rank, submitted_at
                FROM submissions
                ORDER BY avg_rank ASC, submitted_at DESC
                LIMIT 200
                """
            ).fetchall()
        
        entries = [
            {
                "name": r["name"],
                "score_dataset1": float(r["score_dataset1"]),
                "score_dataset2": float(r["score_dataset2"]),
                "score_dataset3": float(r["score_dataset3"]),
                "score_dataset4": float(r["score_dataset4"]),
                "avg_rank": float(r["avg_rank"]),
                "submitted_at": r["submitted_at"],
            }
            for r in rows
        ]
        
        await websocket.send_json({"entries": entries})
        
        # Keep connection alive and handle any incoming messages (if needed)
        while True:
            try:
                # Wait for any message (or just keep connection open)
                data = await websocket.receive_text()
                # Echo back or handle ping/pong if needed
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)
