# Recovery and rollback

1. Call `Control(path).stop()` to persist the stop flag. New steps, claims,
   approvals and releases are rejected. An in-flight SQLite transaction commits
   or rolls back before stop acquires its write lock.
2. Preserve the database and its WAL/SHM files together, or use SQLite's backup
   API. Do not copy only the main file while writers are active.
3. Restart the controller with the same policy version and database. Inspect
   the run and its last evidence event. Use `start()` only when ready to resume.
4. Resume with `run(run_id, owner)`. Execution and its stage commit atomically;
   an interruption cannot create a second staged artifact. Release retries use
   the same workflow/operation key and return the original receipt.
5. Failed runs remain immutable terminal records. `retry(parent, new_run_id,
   fresh_inputs, owner=...)` starts a new attempt at VISIBLE, up to the retry cap.
6. READ_BACK or VERIFIED failure removes staged artifacts transactionally.
   Unapproved/failed/shadow runs cannot release. This is pre-release rollback.
7. For code rollback, check out the prior immutable release tag on a deployment
   checkout after stopping the service. This release uses migration version 1;
   do not downgrade across incompatible future migrations without backup restore.
8. The fixture sink is a local table. Real irreversible effects need their own
   idempotency receipt, authoritative read-back, reconciliation and compensation
   protocol before activation. Database atomicity alone cannot make a remote API
   exactly-once.
