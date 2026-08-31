# Incomplete F4 freeform diagnostic

This third instrumented attempt reproduced the prior two quiet Isaac exits.
The process progressed beyond step 1900, then recorded a normal quick framework
shutdown without a Python traceback, Xid, OOM, or crash record.  The last
flushed trace contains steps 1 through 1750 and remains finite.

This attempt is retained as the representative failed `freeform_s0_cross`
result.  It is not a completed pass and must fail Gate F4 closed.
