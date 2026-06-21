#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

usage() {
    cat <<'USAGE'
Usage: ./deploy.sh [command]

Commands:
  up        Build and start backend + nginx, then check /health (default)
  restart   Restart services and check /health
  stop      Stop services
  logs      Follow service logs
  status    Show service status
  health    Check the public /health endpoint
USAGE
}

detect_compose() {
    if docker compose version >/dev/null 2>&1; then
        COMPOSE=(docker compose)
        return
    fi

    if command -v docker-compose >/dev/null 2>&1; then
        COMPOSE=(docker-compose)
        return
    fi

    echo "Docker Compose is required. Install the docker compose plugin or docker-compose." >&2
    exit 1
}

require_runtime_files() {
    if [[ ! -f .env ]]; then
        echo "Missing .env. Copy .env.example to .env and fill production values first." >&2
        exit 1
    fi

    if [[ ! -f auth/allowlist.yaml ]]; then
        echo "Missing auth/allowlist.yaml. Copy auth/allowlist.example.yaml and add allowed phones first." >&2
        exit 1
    fi
}

compose() {
    "${COMPOSE[@]}" "$@"
}

env_value() {
    local key="$1"
    local value=""

    if [[ -f .env ]]; then
        value="$(sed -n "s/^${key}=//p" .env | tail -n 1)"
    fi

    printf '%s' "$value"
}

health_url() {
    local port="${HTTP_PORT:-}"
    if [[ -z "$port" ]]; then
        port="$(env_value HTTP_PORT)"
    fi
    port="${port:-80}"

    printf 'http://127.0.0.1:%s/health' "$port"
}

wait_for_health() {
    local url

    if ! command -v curl >/dev/null 2>&1; then
        echo "curl is required for health checks." >&2
        exit 1
    fi

    url="$(health_url)"

    echo "Checking health endpoint: $url"
    for _ in $(seq 1 30); do
        if curl -fsS "$url" >/dev/null; then
            echo "Health check passed."
            return
        fi
        sleep 2
    done

    echo "Health check failed. Recent logs:" >&2
    compose logs --tail=80 backend nginx >&2
    exit 1
}

main() {
    local command="${1:-up}"

    if [[ "$command" == "-h" || "$command" == "--help" ]]; then
        usage
        exit 0
    fi

    if ! command -v docker >/dev/null 2>&1; then
        echo "Docker is required." >&2
        exit 1
    fi

    detect_compose

    case "$command" in
        up)
            require_runtime_files
            compose up -d --build
            wait_for_health
            ;;
        restart)
            require_runtime_files
            compose up -d --build
            compose restart backend nginx
            wait_for_health
            ;;
        stop)
            compose down
            ;;
        logs)
            compose logs -f --tail=200
            ;;
        status)
            compose ps
            ;;
        health)
            wait_for_health
            ;;
        *)
            usage >&2
            exit 1
            ;;
    esac
}

main "$@"
