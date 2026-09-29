"""Entry point of the frozen server that build_server.sh builds.

Runs server/server.py against the engine bundled beside it. The PDF document
worker re-invokes this executable as `<exe> document-worker`, because a frozen
program has no interpreter to run server/document_worker.py with.
"""

import sys
from pathlib import Path

if sys.argv[1:2] == ["document-worker"]:
    del sys.argv[1]
    from server import document_worker

    document_worker.main()
else:
    from server import server

    # Before the caller's arguments, so an explicit --binary still wins.
    engine = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "engine/splash"
    sys.argv[1:1] = ["--binary", str(engine)]
    server.main()
