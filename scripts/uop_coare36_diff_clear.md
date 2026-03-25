]633;E;{   echo '# Clear Diff Summary'\x3b   echo\x3b   echo 'Compared files:'\x3b   echo "- $orig"\x3b   echo "- $new"\x3b   echo\x3b   echo '## Function counts'\x3b   echo "- Functions in original: $(wc -l < "$orig_defs")"\x3b   echo "- Functions in revised: $(wc -l < "$new_defs")"\x3b   echo "- Common function names: $(wc -l < "$common_defs")"\x3b   echo\x3b   echo '## Added functions (in revised only)'\x3b   if [[ -s "$added_defs" ]]\x3b then sed 's/^/- /' "$added_defs"\x3b else echo '- None'\x3b fi\x3b   echo\x3b   echo '## Removed functions (missing from revised)'\x3b   if [[ -s "$removed_defs" ]]\x3b then sed 's/^/- /' "$removed_defs"\x3b else echo '- None'\x3b fi\x3b   echo\x3b   echo '## Revised functions (detected in diff hunks)'\x3b   if [[ -s "$changed_defs" ]]\x3b then sed 's/^/- /' "$changed_defs"\x3b else echo '- No function-level hunk headers detected'\x3b fi\x3b   echo\x3b   echo '## Full unified diff'\x3b   echo "See: $patch"\x3b } > "$report";91d6dac8-311d-411e-b325-59e042fac2ad]633;C# Clear Diff Summary

Compared files:
- scripts/coare36original.py
- src/uop_coare/coare36vn_zrf_et.py

## High-level stats
- Functions in original: 10
- Functions in revised: 14
- Added lines: 214
- Deleted lines: 24

## Added functions (in revised only)
- calculate_jd
- calculate_wind_stress_components
- main
- _plot_shortwave_jd_diagnostics

## Removed functions (missing from revised)
- None

## Revised existing functions (detected from diff hunk headers)
- albedo_vector
- coare36vn_zrf_et

## Full patch output
- scripts/uop_coare36_diff_unified.patch

## Useful command for future comparisons
diff -u -p scripts/coare36original.py src/uop_coare/coare36vn_zrf_et.py > scripts/uop_coare36_diff_unified.patch
