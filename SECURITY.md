# Security and data handling

Mandatory examples use offline synthetic data. The package does not read private
electricity-market data or contact external services. Exports store labeled
numeric effects and audit metadata, not original raw observations or model objects.
Source labels may themselves be sensitive: choose shareable labels before export.

Persistence uses JSON and numeric/string NPZ arrays with `allow_pickle=False`.
Do not load pickle files as study results. Treat downloaded archives with normal
resource precautions; the loader validates schemas, not arbitrary file size limits.

Report suspected defects to the project owner through your existing private
channel. A public security contact has not been supplied. Do not invent one.
