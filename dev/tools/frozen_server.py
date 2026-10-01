"""Entry point of the frozen server that build_server.sh builds.

It takes the arguments of `splash serve` (install/launcher.py) at the top
level: `splash-server --model <owner/repo or DIR> [options]`. A frozen
program has no interpreter to run Splash's other Python programs with, so
the launcher runs them in this executable, by a first argument naming one
(install.paths.program):

  install-models ...     install/models.py, the model installer
  server ...             server/server.py against the bundled engine
  catalog-refresh ...    install/catalog.py's background refresh
  document-worker ...    server/document_worker.py, the PDF worker

The launcher's options all start with --, so none is mistaken for one.
"""

import sys
from pathlib import Path

PROGRAMS = ("install-models", "server", "catalog-refresh", "document-worker")


def run_server():
    from server import server

    # Before the caller's arguments, so an explicit --binary still wins.
    engine = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "engine/splash"
    sys.argv[1:1] = ["--binary", str(engine)]
    server.main()


def main():
    program = sys.argv[1] if sys.argv[1:2] and sys.argv[1] in PROGRAMS else None
    if program is None:
        from install import launcher

        return launcher.main(["serve", *sys.argv[1:]])
    del sys.argv[1]
    if program == "install-models":
        from install import models

        return models.main()
    if program == "catalog-refresh":
        from install import catalog

        return catalog.main()
    if program == "document-worker":
        from server import document_worker

        document_worker.main()
        return 0
    run_server()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
