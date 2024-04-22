# AUTO DEV INSTALL

export PATH="$HOME/.rye/shims:$PATH"

cd /app
echo "ACTIVATE: activate .venv"
source .venv/bin/activate

pip install -e .
