set -euo pipefail

if getent group docker >/dev/null && ! id -nG | tr ' ' '\n' | grep -qx docker && command -v sg >/dev/null; then
  exec sg docker -c "cd $(pwd) && bash $0"
fi

export AIRFLOW_HOME=~/airflow
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=false

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

export AIRFLOW__API__PORT=${AIRFLOW_PORT:-8080}
export AIRFLOW__WEBSERVER__WEB_SERVER_PORT=${AIRFLOW_PORT:-8080}

mkdir -p $AIRFLOW_HOME

echo '{"admin": "admin"}' > $AIRFLOW_HOME/simple_auth_manager_passwords.json.generated

uv tool run --with mlflow --with boto3 --with graphviz apache-airflow standalone
