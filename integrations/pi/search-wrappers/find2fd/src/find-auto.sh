#!/usr/bin/env bash
# Measured automatic family: sparse name searches on large, layered trees.
# This gate only observes directory metadata and a bounded, read-only sample.
# It never evaluates a user action or emits user-visible search output.
_find2fd_auto_eligible() (
	local _root=. _pattern='' _kind='' _flag _value _links _maxdepth='' _seen_predicate=0 _output=print
	local _min_depth=0 _sample _sample_total _sample_matches _estimate=0 _count=0 _dir _aggregate_matches=0 _sampled_parents=0
	local -a _dirs=() _sample_args=()
	shopt -s nullglob
	shopt -u failglob nocaseglob
	[[ ${FIND2FD_ACCELERATION-} != off && ${FIND2FD_ACCELERATION-} != 0 ]] || return 1
	# Existing explicit experiment values retain their original opt-in contract.
	[[ ! ${FIND2FD_EXPERIMENTAL_FD+x} ]] || return 1
	[[ -e /proc/"$BASHPID"/fd/0 && -e /proc/"$BASHPID"/fd/1 && -e /proc/"$BASHPID"/fd/2 ]] || return 1
	[[ ${1-} != -P ]] || shift
	if (($#)) && [[ $1 != -* && $1 != '!' && $1 != '(' && $1 != ')' && $1 != ',' ]]; then
		_root=$1
		shift
	fi
	# Byte-safe roots and star-only basename globs use the existing pure engine.
	# Unusual path spelling, raw bytes, following and locale-sensitive patterns
	# go straight to GNU, before Node or sampling.
	[[ $_root && $_root != *//* && $_root != */../* && $_root != */./* && $_root != .. && $_root != ../* && $_root != */.. && $_root != */. ]] || return 1
	local LC_ALL=C
	local _nonascii=$'[^\001-\177]'
	[[ ! $_root =~ $_nonascii ]] || return 1
	while (($#)); do
		_flag=$1
		shift
		case $_flag in
		-name)
			[[ $# -gt 0 && ! $_pattern ]] || return 1
			_pattern=$1
			shift
			[[ $_pattern && ${#_pattern} -le 8192 && $_pattern != *'['* && $_pattern != *']'* && $_pattern != *'{'* && $_pattern != *'}'* && $_pattern != *\\* && $_pattern != *'?'* && $_pattern != *'/'* && ! $_pattern =~ $_nonascii ]] || return 1
			_seen_predicate=1
			;;
		-type)
			[[ $# -gt 0 && ! $_kind && $1 == [fdl] ]] || return 1
			_kind=$1
			shift
			_seen_predicate=1
			;;
		-mindepth | -maxdepth)
			[[ $_seen_predicate == 0 && $# -gt 0 && $1 =~ ^[0-9]{1,6}$ ]] || return 1
			_value=$((10#$1))
			shift
			if [[ $_flag == -maxdepth ]]; then _maxdepth=$_value; else _min_depth=$_value; fi
			;;
		-a | -and) [[ $_seen_predicate == 1 && $# -gt 0 ]] || return 1 ;;
		-print) [[ $# == 0 && ! -t 1 ]] || return 1 ;;
		-print0)
			[[ $# == 0 ]] || return 1
			_output=print0
			;;
		*) return 1 ;;
		esac
	done
	[[ $_pattern && (-z $_maxdepth || $_maxdepth -ge 3) ]] || return 1
	[[ $_output == print0 || ! -t 1 ]] || return 1
	[[ $_min_depth -le 3 && -d $_root && ! -L $_root ]] || return 1
	# Link counts identify sufficiently large layered directory trees without
	# recursively enumerating them. Unknown/non-counting filesystems stay GNU.
	_links=$(command stat -c %h -- "$_root" 2>/dev/null) || return 1
	[[ $_links =~ ^[0-9]+$ && $_links -ge 34 && $_links -le 514 ]] || return 1
	_dirs=("$_root"/*/ "$_root"/.*/)
	for _dir in "${_dirs[@]}"; do
		[[ $_dir != "$_root/./" && $_dir != "$_root/../" && -d $_dir && ! -L ${_dir%/} ]] || continue
		_dirs[_count]=$_dir
		((_count += 1))
		[[ $_count -lt 8 ]] || break
	done
	[[ $_count -gt 0 ]] || return 1
	_dirs=("${_dirs[@]:0:_count}")
	local _observed=0
	while read -r _links; do
		[[ $_links =~ ^[0-9]+$ && $_links -le 10000 ]] || return 1
		((_observed += 1))
		((_estimate += _links > 2 ? _links - 2 : 0))
	done < <(command stat -c %h -- "${_dirs[@]}" 2>/dev/null)
	[[ $_observed == "$_count" ]] || return 1
	# The sampled top-level parents represent at least 24K second-level dirs.
	_links=$(command stat -c %h -- "$_root" 2>/dev/null) || return 1
	((_estimate * (_links - 2) >= 24000 * _count)) || return 1
	_sample_args=(-maxdepth 2 -printf x)
	[[ -z $_kind ]] || _sample_args+=(-type "$_kind")
	_sample_args+=(-name "$_pattern" -printf y)
	# Up to eight bounded samples catch dense names and repeated filenames.
	# Internal SIGPIPE after the sample budget is expected and never escapes.
	for _dir in "${_dirs[@]}"; do
		if ! _sample=$(
			set +o pipefail
			command find "$_dir" "${_sample_args[@]}" 2>/dev/null | command head -c 32768
			_value=${PIPESTATUS[0]}
			[[ $_value == 0 || $_value == 141 ]]
		); then return 1; fi
		_value=${_sample//[^y]/}
		_sample_matches=${#_value}
		_sample_total=$((${#_sample} - _sample_matches))
		((_sample_total >= 128 && ${#_sample} < 32768 && _sample_matches * 128 <= _sample_total)) || return 1
		((_aggregate_matches += _sample_matches))
		((_sampled_parents += 1))
	done
	# Repeated names can look proportionally sparse but require reading nearly
	# every ordering parent. Cap the extrapolated candidate count as well.
	((_aggregate_matches * (_links - 2) <= 32 * _sampled_parents)) || return 1
	return 0
)
