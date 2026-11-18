# Exports the variables of .env that are set (an empty one keeps its default), ~/ expanded. uv reads
# where it puts the virtualenv, its cache and its Pythons from the environment only, never from .env:
# source this file in each new shell (and in each job script) before calling uv.
#
#     source env.sh            # bash or zsh, from any folder; then uv sync, uv run ... as usual
#
# A variable already set in the shell wins over .env, as in paths.py.

_tsc_env="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/.env"
if [ -f "$_tsc_env" ]; then
    while IFS= read -r _tsc_line || [ -n "$_tsc_line" ]; do
        case "$_tsc_line" in *=*) ;; *) continue ;; esac
        _tsc_key="${_tsc_line%%=*}"
        case "$_tsc_key" in ''|[0-9]*|*[!A-Za-z0-9_]*) continue ;; esac   # comments, malformed lines
        _tsc_value="${_tsc_line#*=}"
        case "$_tsc_value" in \"*\"|\'*\') _tsc_value="${_tsc_value#?}"; _tsc_value="${_tsc_value%?}" ;; esac
        case "$_tsc_value" in '~/'*) _tsc_value="$HOME/${_tsc_value#??}" ;; esac
        if [ -n "$_tsc_value" ] && [ -z "$(printenv "$_tsc_key")" ]; then
            export "$_tsc_key=$_tsc_value"
        fi
    done < "$_tsc_env"
else
    echo "env.sh: no $_tsc_env" >&2
fi
unset _tsc_env _tsc_line _tsc_key _tsc_value
