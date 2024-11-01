"""Print SHA256 hashes of files to be used in phasorpy.datasets.

::

    $ python sha256.py [files or directory or glob-pattern]

"""

import os
import sys
from glob import glob
from hashlib import sha256

from tifffile import natural_sorted

nargs = len(sys.argv)
if nargs == 1:
    files = natural_sorted(glob('*.*'))
elif nargs == 2:
    arg = sys.argv[1]
    if '*' in arg or '?' in arg:
        files = natural_sorted(glob(arg))
    elif os.path.isdir(arg):
        files = natural_sorted(glob(os.path.join(arg, '*.*')))
    else:
        files = [arg]
else:
    files = list(sys.argv[1:])

print()
for fname in files:
    with open(fname, 'rb') as fh:
        data = fh.read()
    sha = sha256(data).hexdigest()
    fname = os.path.split(fname)[-1]
    print(f"    {fname!r}: (\n        'sha256:'\n        '{sha}'\n    ),")
