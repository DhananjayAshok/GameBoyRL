# Bash Templates

Reference for bash scripting patterns used in this project family.

---

## Quick Reference

| Task | Function |
|------|----------|
| Serialize ARGS dict to `--key value` flags | `args_to_flags ARGS` |
| Serialize only a subset of keys | `args_to_flags_subset ARGS KEYS_ARRAY` |
| Get a computed string (e.g. exp name) from Python | `get_string_from_args <name> ARGS` |
| Copy all keys from one dict to another | `populate_dict SRC DST` |
| Append all elements from one array to another | `populate_array SRC DST` |
| Copy a subset of keys between dicts | `populate_dict_subset SRC DST KEYS` |

All functions use bash namerefs — pass variable **names**, not values (no `$`).

---

## Core Rules

### 1. `none` is the sentinel for "absent" — never `""`

All optional args that mean "not provided" must default to `none`, not `""`. This is enforced two ways:
- The boilerplate usage-string loop will `exit 1` if any optional default is blank.
- `args_to_flags` silently converts `""` → `"none"` anyway, but explicit `none` makes the intent clear.

**Consequence:** every script that receives a `none` value must handle it explicitly. Check before using as a path, before forwarding a flag, etc.:
```bash
if [[ "${ARGS["scheduler"]}" != "none" ]]; then
    scheduler_arg="--scheduler ${ARGS["scheduler"]}"
else
    scheduler_arg=""
fi
```

### 2. Bool flags must be converted — never passed as `--flag true`

Python click `is_flag=True` options (`--overwrite`, `--verbose`, etc.) don't accept `--flag true`. Store them as the string `false`/`true` in ARGS, then convert explicitly:
```bash
if [[ "${ARGS["verbose"]}" == "true" || "${ARGS["verbose"]}" == "yes" || "${ARGS["verbose"]}" == "y" ]]; then
    verbose_flag="--verbose"
else
    verbose_flag=""
fi
# Then pass $verbose_flag, not via args_to_flags
```

### 3. Defaults must never be blank

The boilerplate usage-string loop catches this at startup:
```bash
if [[ -z "${ARGS[$opt]}" ]]; then
    echo "DEFAULT VALUE OF KEY \"$opt\" CANNOT BE BLANK"; exit 1
fi
```
If an arg is genuinely optional at the Python level, its bash default is `none` (see rule 1).

---

## Script Boilerplate

Every script follows this structure exactly. The argument-parsing block is **copy-verbatim**.

```bash
#!/usr/bin/env bash

source scripts/utils.sh || { echo "Could not source utils"; exit 1; }

# Script-specific defaults and required args.
# Populate from utils.sh shared groups BEFORE setting ALLOWED_FLAGS.
declare -A ARGS
REQUIRED_ARGS=()

populate_dict SOME_DEFAULTS ARGS          # optional args with defaults
populate_array SOME_ESSENTIALS REQUIRED_ARGS  # required args (no defaults)

# Script-local additions after populate_dict:
ARGS["my_extra_arg"]="default_val"

# --- Argument parsing (copy verbatim) ---
ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}" "${!ARGS[@]}")
USAGE_STR="Usage: $0"
for req in "${REQUIRED_ARGS[@]}"; do
    USAGE_STR+=" --$req <value>"
done
for opt in "${!ARGS[@]}"; do
    if [[ ! " ${REQUIRED_ARGS[*]} " =~ " ${opt} " ]]; then
        if [[ -z "${ARGS[$opt]}" ]]; then
            echo "DEFAULT VALUE OF KEY \"$opt\" CANNOT BE BLANK"; exit 1
        fi
        USAGE_STR+=" [--$opt <value> (default: ${ARGS[$opt]})]"
    fi
done
function usage() { echo "$USAGE_STR"; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
            FLAG=${1#--}
            VALID=false
            for allowed in "${ALLOWED_FLAGS[@]}"; do
                if [[ "$FLAG" == "$allowed" ]]; then VALID=true; break; fi
            done
            if [ "$VALID" = false ]; then echo "Error: Unknown flag --$FLAG"; usage; fi
            ARGS["$FLAG"]="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

for req in "${REQUIRED_ARGS[@]}"; do
    if [[ -z "${ARGS[$req]}" ]]; then echo "Error: --$req is required."; FAILED=true; fi
done
if [ "$FAILED" = true ]; then usage; fi
# --- End argument parsing ---

# Print active variables
echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

# Script logic below:
```

---

## Shared Args Pattern (utils.sh)

Every command context has three things defined in `utils.sh`:

```bash
# 1. Essentials array — required args, no defaults
SOME_ESSENTIALS=("game" "init_state")
populate_array ESSENTIAL_ARGS SOME_ESSENTIALS  # ESSENTIAL_ARGS=("game") is always included

# 2. Defaults dict — optional args, all non-empty
declare -A SOME_DEFAULTS=(
    ["overwrite"]=false
    ["verbose"]=false
    ["path_arg"]=none       # absent path/optional → none, never ""
    ["max_steps"]=30
)

# 3. ARG_KEYS array = essentials + defaults keys — used as the subset mask
SOME_ARG_KEYS=("${SOME_ESSENTIALS[@]}" "${!SOME_DEFAULTS[@]}")
```

**Composing from existing groups** — use `populate_dict` / `populate_array` / `populate_dict_subset`:
```bash
declare -A TRAINING_DEFAULTS
populate_dict ENV_DEFAULTS TRAINING_DEFAULTS       # inherit env args
populate_dict ALGORITHM_DEFAULTS TRAINING_DEFAULTS # inherit algorithm args
TRAINING_DEFAULTS["my_extra"]="val"               # script-local addition after populate
TRAINING_ARG_KEYS=("${TRAINING_ESSENTIALS[@]}" "${!TRAINING_DEFAULTS[@]}")
```

**Inheriting a subset** — when a new group only wants some keys from another:
```bash
SAME_AS_TRAINING=("observation_embedder" "embedder_load_path")
declare -A MY_DEFAULTS
populate_dict_subset TRAINING_DEFAULTS MY_DEFAULTS SAME_AS_TRAINING
```

---

## Calling Subscripts

Always use `args_to_flags_subset` + the subscript's `_ARG_KEYS` to forward only what it accepts:

```bash
arg_string=$(args_to_flags_subset ARGS TRAINING_ARG_KEYS)
bash scripts/train.sh $arg_string
```

When looping and mutating ARGS in-place (e.g. iterating over init states), set the key on ARGS before the call:
```bash
IFS=',' read -ra init_states_arr <<< "${TRAIN_STATES[$game]}"
for init_state in "${init_states_arr[@]}"; do
    ARGS["init_state"]=$init_state
    arg_string=$(args_to_flags_subset ARGS SOME_ARG_KEYS)
    bash scripts/subscript.sh $arg_string
done
```

---

## "All" Script Header

Scripts that loop over all init states for a game always start with this exact header:

```bash
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
python scripts/create_task_dictionary.py
source scripts/all_train_states.sh
```

Then loop via `TRAIN_STATES[$game]` (comma-separated):
```bash
game="${ARGS["game"]}"
IFS=',' read -ra init_states_arr <<< "${TRAIN_STATES[$game]}"
for init_state in "${init_states_arr[@]}"; do
    ARGS["init_state"]=$init_state
    arg_string=$(args_to_flags_subset ARGS SOME_ARG_KEYS)
    bash scripts/some_script.sh $arg_string
done
```

When the "all" script doesn't require `init_state` as an input arg (it comes from the loop), set `REQUIRED_ARGS` manually without it — don't use `populate_array SOME_ESSENTIALS REQUIRED_ARGS` if that array includes `init_state`.

---

## `scripts/get_strings.py`: Python String Generation

For computed strings (experiment names, paths) that are easier to build in Python:

```bash
exp_name=$(get_string_from_args exp_name ARGS)
if [[ -z "$exp_name" ]]; then echo "Error: failed to build exp name"; exit 1; fi
model_save_path="$storage_dir/models/$exp_name/"
```

`none` values in ARGS are passed as `None` to the Python function.

### Adding a new string function (in `scripts/get_strings.py`)

```python
class MyExpName(StringFunction):
    NAME = "my_exp_name"
    REQUIRED_ARGS = ["dataset", "model"]
    OPTIONAL_ARGS = {"version": "v1"}

    def _get_string(self, **kwargs) -> str:
        return f"{kwargs['dataset']}_{kwargs['model']}_{kwargs['version']}"

STRING_FUNCTIONS = [..., MyExpName]
```

Rules:
- `_get_string` must return a string with **no spaces** if used as a path
- Unexpected arguments are silently ignored — keep `REQUIRED_ARGS` minimal
- `none` values arrive as `None` in kwargs; handle or ignore as needed
