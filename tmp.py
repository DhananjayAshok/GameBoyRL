import numpy as np, pandas as pd
from create_dataset import select_successful

# 1. binary mode: equals the old success==True selection
b = pd.DataFrame({"success": [True, False, True], "score": [np.nan, np.nan, np.nan]})
sel = select_successful(b, 6.0)
assert list(sel.index) == [0, 2], list(sel.index)
assert list(sel.index) == list(b[b["success"] == True].index)  # backward compatible

# 2. score mode: success all NaN, threshold applies
s = pd.DataFrame({"success": [np.nan, np.nan, np.nan], "score": [7.0, 5.0, 6.0]})
assert list(select_successful(s, 6.0).index) == [0, 2], list(select_successful(s, 6.0).index)
assert list(select_successful(s, 8.0).index) == [], "threshold 8 should drop all"

# 3. mixed rows (robust even if a csv somehow has both)
m = pd.DataFrame({"success": [True, np.nan, False, np.nan], "score": [np.nan, 8.0, np.nan, 4.0]})
assert list(select_successful(m, 6.0).index) == [0, 1], list(select_successful(m, 6.0).index)

# 4. clean_practice imports the SAME function object (no drift possible).
# NB: vlm_scripts/__init__ binds the name `clean_practice` to the Command, which
# shadows the submodule as a package attribute, so fetch the real module module
# object from sys.modules instead of via attribute access.
import sys, importlib
importlib.import_module("vlm_scripts.clean_practice")
cp = sys.modules["vlm_scripts.clean_practice"]
assert cp.select_successful is select_successful
from create_dataset import _episode_cutoff as cd_cut
assert cp._episode_cutoff is cd_cut

print("SELECT_SUCCESSFUL_OK; clean_practice shares the same selector+cutoff")
