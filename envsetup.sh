# AUTO DEV SETUP

# check if rye is installed
if ! command -v rye &>/dev/null; then
    echo "rye could not be found: installing now ..."
    curl -sSf https://rye-up.com/get | RYE_INSTALL_OPTION="--yes" bash
    echo "Check the rye docs for more info: https://rye-up.com/"
fi

export PATH="$HOME/.rye/shims:$PATH"

cd /app
source "$HOME/.rye/env"
echo 'source "$HOME/.rye/env"' >> ~/.bashrc
echo 'source "$HOME/.rye/env"' >> ~/.profile


echo "SYNC: setup .venv"
rye sync

rye add pytest pre-commit

echo "ACTIVATE: activate .venv"
source .venv/bin/activate

echo "SETUP: install pre-commit hooks"
pre-commit install
