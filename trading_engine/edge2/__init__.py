"""EDGE2 gap/catalyst scanner, ported as an IronFrost discovery layer.

Ported modules (scanner, universe, catalyst_nlp, instrument_filter,
outcome_tracker, database) keep their original EDGE2 logic; the port only
rewires imports, makes the SQLite path configurable (one DB: IronFrost's
trading.db), and adds one seam in scanner.scan_universe(): the first flag
of the day per ticker opens an IronFrost PAPER order via bridge.py, tagged
source='edge2' end to end. The EDGE2 Flask app and templates were dropped
— IronFrost's engine loop and dashboards take over.

CatalystNLPProcessor stays in keyword mode (enable_model=False); torch and
transformers are deliberately NOT dependencies.
"""
