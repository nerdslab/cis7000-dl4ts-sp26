#!/bin/bash

BACKEND_DIR="/home/smahato/cis7000-dl4ts-sp26/nb2-comp/backend"
DB_FILE="$BACKEND_DIR/leaderboard.db"
BACKUP_DIR="$BACKEND_DIR/backups"

mkdir -p "$BACKUP_DIR"

last_backup_file=""

while true; do
    timestamp=$(date +"%Y%m%d_%H%M%S")
    backup_file="$BACKUP_DIR/leaderboard_${timestamp}.db"

    # Delete previous backup if it exists
    if [[ -n "$last_backup_file" && -e "$last_backup_file" ]]; then
        rm -f "$last_backup_file"
    fi

    cp "$DB_FILE" "$backup_file"
    echo "[$(date)] Backup: ${backup_file}"

    last_backup_file="$backup_file"
    sleep 300
done