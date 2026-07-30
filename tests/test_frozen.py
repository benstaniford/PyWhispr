from unittest.mock import patch

from pywhispr import frozen


class TestFrozenEntryPoint:
    """The packaged exe is also the command line: cuda.verify() re-runs it."""

    def test_no_arguments_runs_the_app(self):
        with patch("pywhispr.app.run_app", return_value=0) as run_app:
            assert frozen.run([]) == 0
        run_app.assert_called_once()

    def test_a_subcommand_runs_the_cli_not_a_second_app(self):
        """It used to ignore argv, so `verify-gpu` started the whole app again --
        a second tray icon, a second model download, a second progress window."""
        with (
            patch("pywhispr.cli.main", return_value=0) as cli,
            patch("pywhispr.app.run_app") as run_app,
        ):
            assert frozen.run(["verify-gpu", "--quantization", ""]) == 0
        cli.assert_called_once_with(["verify-gpu", "--quantization", ""])
        run_app.assert_not_called()

    def test_the_exit_code_is_passed_through(self):
        """cuda.verify() reads it as the verdict."""
        with patch("pywhispr.cli.main", return_value=1):
            assert frozen.run(["verify-gpu"]) == 1
