#!/usr/bin/env bash
#
# push_to_github.sh — one-shot script to publish this project to GitHub.
#
# Usage:
#   1. Edit the two variables below (GH_USER and REPO_NAME).
#   2. Make it executable:  chmod +x push_to_github.sh
#   3. Run it:              ./push_to_github.sh
#
# Requires: git. Optionally the GitHub CLI `gh` (auto-detected).
# --------------------------------------------------------------------

set -e  # stop on first error

# ---- EDIT THESE ----
GH_USER="your-username"
REPO_NAME="rag-assistant"
VISIBILITY="public"      # or "private"
# --------------------

echo "==> Initialising git repository..."
git init

echo "==> Adding files..."
git add .

echo "==> Creating first commit..."
git commit -m "Initial commit: RAG Document Assistant (Claude + FAISS + Streamlit)"

echo "==> Setting branch to main..."
git branch -M main

# If the GitHub CLI is installed, use it to create + push in one step.
if command -v gh >/dev/null 2>&1; then
  echo "==> GitHub CLI detected. Creating remote repo and pushing..."
  gh repo create "$REPO_NAME" --"$VISIBILITY" --source=. --remote=origin --push
else
  echo "==> GitHub CLI not found."
  echo "    Create an EMPTY repo named '$REPO_NAME' at https://github.com/new"
  echo "    (do NOT add a README or .gitignore there), then press Enter to continue."
  read -r _
  git remote add origin "https://github.com/$GH_USER/$REPO_NAME.git"
  git push -u origin main
fi

echo ""
echo "Done. Your repo is live at: https://github.com/$GH_USER/$REPO_NAME"
