"""Third-party code vendored into SpecMod.

Everything here was written by someone else and is redistributed under its own
licence, which is reproduced in the module that carries it. The rules for this
package:

1. **Nothing in here is public API.** Callers go through the wrapper in
   :mod:`specmod.transforms`, so a vendored implementation can be replaced by a
   native one without anybody noticing.
2. **Changes are recorded.** Each module lists what was altered from upstream
   and why, so the diff against the original stays legible.
3. **Our tests own it.** Vendored code is held to the same contracts as
   everything else — a vendored function that cannot satisfy the Parseval check
   in :mod:`specmod.core.spectrum` is a bug on our side of the fence now.

The point of the quarantine is that upstream code arrives with upstream
conventions, and mixing those into ``specmod.transforms`` is how a codebase
ends up with two of everything.
"""

from __future__ import annotations
