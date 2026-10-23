import os
import datetime
from typing import Dict, Any

try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


class MetricsDB:
    """PostgreSQL client for storing Evidently drift detection metrics and alerting."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        db_name: str = "mlflow",
        user: str = "postgres",
        password: str = "postgres",
    ):
        self.host = os.getenv("POSTGRES_HOST", host)
        self.port = int(os.getenv("POSTGRES_PORT", str(port)))
        self.db_name = os.getenv("POSTGRES_DB", db_name)
        self.user = os.getenv("POSTGRES_USER", user)
        self.password = os.getenv("POSTGRES_PASSWORD", password)
        self._init_db()

    def _get_connection(self):
        if not PSYCOPG2_AVAILABLE:
            return None
        try:
            return psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.db_name,
                user=self.user,
                password=self.password,
                connect_timeout=3,
            )
        except Exception:
            return None

    def _init_db(self):
        conn = self._get_connection()
        if not conn:
            return

        create_table_sql = """
        CREATE TABLE IF NOT EXISTS drift_metrics (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            dataset_name VARCHAR(100) NOT NULL,
            drift_score FLOAT NOT NULL,
            number_of_drifted_columns INT NOT NULL,
            share_of_drifted_columns FLOAT NOT NULL,
            dataset_drift BOOLEAN NOT NULL,
            retraining_recommended BOOLEAN NOT NULL
        );
        """
        try:
            with conn.cursor() as cur:
                cur.execute(create_table_sql)
            conn.commit()
            conn.close()
            print("[MetricsDB] PostgreSQL drift_metrics table verified.")
        except Exception as e:
            print(f"[MetricsDB] Notice: Could not create table: {e}")

    def log_drift_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Persist computed drift metrics into PostgreSQL table."""
        ts = metrics.get("timestamp", datetime.datetime.now(datetime.timezone.utc))
        dataset = metrics.get("dataset_name", "march-2023-live")
        drift_score = float(metrics.get("drift_score", 0.0))
        n_drifted = int(metrics.get("number_of_drifted_columns", 0))
        share_drifted = float(metrics.get("share_of_drifted_columns", 0.0))
        dataset_drift = bool(metrics.get("dataset_drift", False))
        retrain = bool(metrics.get("retraining_recommended", False))

        conn = self._get_connection()
        if conn:
            insert_sql = """
            INSERT INTO drift_metrics (
                timestamp, dataset_name, drift_score, 
                number_of_drifted_columns, share_of_drifted_columns, 
                dataset_drift, retraining_recommended
            ) VALUES (%s, %s, %s, %s, %s, %s, %s);
            """
            try:
                with conn.cursor() as cur:
                    cur.execute(insert_sql, (ts, dataset, drift_score, n_drifted, share_drifted, dataset_drift, retrain))
                conn.commit()
                conn.close()
                print(f"[MetricsDB] Successfully logged drift metrics to PostgreSQL (drift_share={share_drifted:.2%}).")
                return True
            except Exception as e:
                print(f"[MetricsDB] Failed to insert drift row: {e}")
                return False
        else:
            print(f"[MetricsDB] (Local Record) drift_score={drift_score:.4f}, retrain_recommended={retrain}")
            return True
