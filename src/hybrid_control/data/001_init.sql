CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (run_id TEXT NOT NULL, seq INTEGER NOT NULL,
                    data TEXT NOT NULL, PRIMARY KEY(run_id,seq));
                CREATE TABLE IF NOT EXISTS staged (run_id TEXT PRIMARY KEY, artifact TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS effects (workflow_id TEXT NOT NULL, effect_key TEXT NOT NULL,
                    artifact TEXT NOT NULL, receipt TEXT NOT NULL, PRIMARY KEY(workflow_id,effect_key));
                CREATE TABLE IF NOT EXISTS receipts (receipt_id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT OR IGNORE INTO settings VALUES ('stopped','false');
                PRAGMA user_version=1;
