"""install/paths.py: the data directory and the layout beneath it."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from install import launcher, paths


class LayoutTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_a_custom_data_directory_holds_everything(self):
        layout = paths.layout(self.root / "data")
        data = (self.root / "data").absolute()
        self.assertEqual(layout.data, data)
        self.assertEqual(layout.models, data / "models")
        self.assertEqual(layout.runtime, data / "runtime")
        self.assertEqual(layout.profiles, data / "runtime")
        self.assertEqual(layout.weights, data / "weights")
        self.assertEqual(layout.hub_cache, data / "huggingface")
        self.assertEqual(layout.cache, data / "runtime/cache")
        with mock.patch.dict(
            os.environ,
            {paths.DATA_VARIABLE: str(self.root / "env"), "HOME": str(self.root)},
        ):
            self.assertEqual(paths.layout().data, self.root / "env")
            self.assertEqual(paths.layout("~/tilde").data, self.root / "tilde")
            self.assertEqual(paths.custom_data_dir(), self.root / "env")
        with mock.patch.dict(os.environ, {paths.DATA_VARIABLE: ""}):
            self.assertIsNone(paths.custom_data_dir())

    def test_the_default_layout_is_the_checkout_or_the_release_directory(self):
        with mock.patch.dict(os.environ, {paths.DATA_VARIABLE: ""}):
            layout = paths.layout()
            self.assertEqual(layout.data, paths.ROOT)
            self.assertEqual(layout.models, paths.ROOT / "install/models")
            self.assertEqual(layout.runtime, paths.ROOT / "build/runtime")
            self.assertEqual(layout.cache, paths.ROOT / "build/runtime/cache")
            self.assertEqual(layout.profiles, paths.ROOT / "install/profiles")
            with mock.patch.object(paths, "PACKAGED", True):
                release = Path.home() / "Library/Application Support/Splash"
                self.assertEqual(paths.layout().models, release / "models")

    def test_the_environment_of_a_custom_data_directory_gathers_the_caches(self):
        layout = paths.layout(self.root / "data")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(launcher.data_environment(layout, custom=False), {})
            self.assertEqual(
                launcher.data_environment(layout, custom=True),
                {
                    paths.DATA_VARIABLE: str(layout.data),
                    "SPLASH_WEIGHT_CACHE": str(layout.weights),
                    "HF_HUB_CACHE": str(layout.hub_cache),
                },
            )
        # What the user set explicitly stands.
        with mock.patch.dict(
            os.environ,
            {"SPLASH_WEIGHT_CACHE": "/weights", "HF_HOME": "/hf"},
            clear=True,
        ):
            self.assertEqual(
                launcher.data_environment(layout, custom=True),
                {paths.DATA_VARIABLE: str(layout.data)},
            )


if __name__ == "__main__":
    unittest.main()


class ProgramTest(unittest.TestCase):
    def test_a_program_runs_under_python_or_as_the_frozen_servers_subcommand(self):
        script = paths.ROOT / "server/server.py"
        with mock.patch.object(paths, "FROZEN", False):
            self.assertEqual(
                paths.program("server", script, "-u"),
                [str(paths.PYTHON), "-u", str(script)],
            )
        with (
            mock.patch.object(paths, "FROZEN", True),
            mock.patch.object(paths.sys, "executable", "/bundle/splash-server"),
        ):
            # Interpreter options have no place: the executable takes none.
            self.assertEqual(
                paths.program("server", script, "-u"),
                ["/bundle/splash-server", "server"],
            )
