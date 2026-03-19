"""OcuTrace GUI Launcher.

wxPython-based launcher that provides a simple interface for clinical
staff: enter patient name, calibrate, run test, view report.

PsychoPy operations run in a child process via multiprocessing to
avoid event loop conflicts with wxPython.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

import wx


# --- Worker functions (run in child process) ---

def _calibration_worker(participant_id: str, result_queue: mp.Queue) -> None:
    """Run calibration in a child process."""
    try:
        from src.orchestrator import run_calibration
        result = run_calibration(participant_id)
        result_queue.put(("success", result))
    except Exception as e:
        result_queue.put(("error", str(e)))


def _experiment_worker(
    participant_id: str,
    calibrate_first: bool,
    result_queue: mp.Queue,
) -> None:
    """Run experiment in a child process."""
    try:
        from src.orchestrator import run_experiment
        result = run_experiment(
            participant_id,
            calibrate_first=calibrate_first,
        )
        result_queue.put(("success", result))
    except Exception as e:
        result_queue.put(("error", str(e)))


# --- Main GUI ---

class LauncherFrame(wx.Frame):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__(
            None,
            title="OcuTrace - Saccade Analysis",
            size=(520, 480),
            style=wx.DEFAULT_FRAME_STYLE & ~wx.RESIZE_BORDER,
        )
        self.SetBackgroundColour(wx.Colour(30, 30, 40))
        self._worker: Optional[mp.Process] = None
        self._result_queue = mp.Queue()
        self._last_session_id: Optional[str] = None

        self._build_ui()
        self._timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_timer, self._timer)
        self.Centre()

    def _build_ui(self) -> None:
        """Construct the UI elements."""
        panel = wx.Panel(self)
        panel.SetBackgroundColour(wx.Colour(30, 30, 40))
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Title
        title = wx.StaticText(panel, label="OcuTrace")
        title.SetForegroundColour(wx.Colour(70, 160, 230))
        title_font = wx.Font(24, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        title.SetFont(title_font)
        sizer.Add(title, 0, wx.ALIGN_CENTER | wx.TOP, 20)

        subtitle = wx.StaticText(panel, label="Webcam-based saccade & antisaccade analysis")
        subtitle.SetForegroundColour(wx.Colour(150, 150, 170))
        sizer.Add(subtitle, 0, wx.ALIGN_CENTER | wx.BOTTOM, 20)

        # Patient name input
        name_sizer = wx.BoxSizer(wx.HORIZONTAL)
        lbl = wx.StaticText(panel, label="Patient Name:")
        lbl.SetForegroundColour(wx.WHITE)
        lbl_font = wx.Font(11, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        lbl.SetFont(lbl_font)
        name_sizer.Add(lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)

        self._name_input = wx.TextCtrl(panel, size=(280, 30))
        self._name_input.SetFont(lbl_font)
        name_sizer.Add(self._name_input, 1, wx.EXPAND)
        sizer.Add(name_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 40)
        sizer.AddSpacer(25)

        # Buttons
        btn_font = wx.Font(12, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)

        self._btn_calibrate = wx.Button(panel, label="Calibrate", size=(200, 45))
        self._btn_calibrate.SetFont(btn_font)
        self._btn_calibrate.SetBackgroundColour(wx.Colour(50, 120, 180))
        self._btn_calibrate.SetForegroundColour(wx.WHITE)
        self._btn_calibrate.Bind(wx.EVT_BUTTON, self._on_calibrate)
        sizer.Add(self._btn_calibrate, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)

        self._btn_test = wx.Button(panel, label="Run Test", size=(200, 45))
        self._btn_test.SetFont(btn_font)
        self._btn_test.SetBackgroundColour(wx.Colour(40, 140, 80))
        self._btn_test.SetForegroundColour(wx.WHITE)
        self._btn_test.Bind(wx.EVT_BUTTON, self._on_test)
        sizer.Add(self._btn_test, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)

        self._btn_cal_and_test = wx.Button(
            panel, label="Calibrate + Test", size=(200, 45)
        )
        self._btn_cal_and_test.SetFont(btn_font)
        self._btn_cal_and_test.SetBackgroundColour(wx.Colour(160, 100, 40))
        self._btn_cal_and_test.SetForegroundColour(wx.WHITE)
        self._btn_cal_and_test.Bind(wx.EVT_BUTTON, self._on_cal_and_test)
        sizer.Add(self._btn_cal_and_test, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)

        self._btn_report = wx.Button(panel, label="Show Report", size=(200, 45))
        self._btn_report.SetFont(btn_font)
        self._btn_report.SetBackgroundColour(wx.Colour(100, 60, 140))
        self._btn_report.SetForegroundColour(wx.WHITE)
        self._btn_report.Bind(wx.EVT_BUTTON, self._on_report)
        sizer.Add(self._btn_report, 0, wx.ALIGN_CENTER | wx.BOTTOM, 20)

        # Status
        self._status = wx.StaticText(panel, label="Ready")
        self._status.SetForegroundColour(wx.Colour(120, 200, 120))
        status_font = wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self._status.SetFont(status_font)
        sizer.Add(self._status, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)

        panel.SetSizer(sizer)

    def _get_patient_name(self) -> Optional[str]:
        """Validate and return patient name."""
        name = self._name_input.GetValue().strip()
        if not name:
            wx.MessageBox(
                "Please enter the patient name.",
                "Missing Name",
                wx.OK | wx.ICON_WARNING,
            )
            return None
        return name

    def _set_busy(self, msg: str) -> None:
        """Disable buttons and show status."""
        self._status.SetForegroundColour(wx.Colour(230, 200, 80))
        self._status.SetLabel(msg)
        self._btn_calibrate.Disable()
        self._btn_test.Disable()
        self._btn_cal_and_test.Disable()
        self._btn_report.Disable()
        self._name_input.Disable()

    def _set_ready(self, msg: str = "Ready", success: bool = True) -> None:
        """Re-enable buttons and show status."""
        color = wx.Colour(120, 200, 120) if success else wx.Colour(230, 80, 80)
        self._status.SetForegroundColour(color)
        self._status.SetLabel(msg)
        self._btn_calibrate.Enable()
        self._btn_test.Enable()
        self._btn_cal_and_test.Enable()
        self._btn_report.Enable()
        self._name_input.Enable()

    def _on_calibrate(self, event: wx.Event) -> None:
        name = self._get_patient_name()
        if not name:
            return
        self._set_busy("Calibrating... (look at the dots)")
        self._worker = mp.Process(
            target=_calibration_worker,
            args=(name, self._result_queue),
        )
        self._worker.start()
        self._timer.Start(300)

    def _on_test(self, event: wx.Event) -> None:
        name = self._get_patient_name()
        if not name:
            return
        self._set_busy("Running test... (follow the dots)")
        self._worker = mp.Process(
            target=_experiment_worker,
            args=(name, False, self._result_queue),
        )
        self._worker.start()
        self._timer.Start(300)

    def _on_cal_and_test(self, event: wx.Event) -> None:
        name = self._get_patient_name()
        if not name:
            return
        self._set_busy("Calibrating + Testing... (follow the dots)")
        self._worker = mp.Process(
            target=_experiment_worker,
            args=(name, True, self._result_queue),
        )
        self._worker.start()
        self._timer.Start(300)

    def _on_report(self, event: wx.Event) -> None:
        name = self._get_patient_name()
        if not name:
            return
        self._set_busy("Generating report...")
        try:
            from src.orchestrator import generate_report
            report_path = generate_report(session_id=self._last_session_id)
            import platform
            if platform.system() == "Windows":
                os.startfile(str(report_path))
            elif platform.system() == "Darwin":
                import subprocess
                subprocess.run(["open", str(report_path)])
            else:
                import subprocess
                subprocess.run(["xdg-open", str(report_path)])
            self._set_ready(f"Report: {report_path.name}")
        except Exception as e:
            self._set_ready(f"Error: {e}", success=False)

    def _on_timer(self, event: wx.Event) -> None:
        """Poll for child process completion."""
        if self._worker is None:
            self._timer.Stop()
            return

        if self._worker.is_alive():
            return

        self._timer.Stop()
        self._worker.join()
        self._worker = None

        try:
            status_type, result = self._result_queue.get_nowait()
        except Exception:
            self._set_ready("Process ended unexpectedly", success=False)
            return

        if status_type == "error":
            self._set_ready(f"Error: {result}", success=False)
        else:
            session_id = result.get("session_id", "")
            self._last_session_id = session_id

            if "metrics" in result and result["metrics"] is not None:
                m = result["metrics"]
                self._set_ready(
                    f"Done! {result['n_trials']} trials | "
                    f"Anti err: {m.antisaccade_error_rate:.0%} | "
                    f"Session: {session_id[:8]}..."
                )
            elif "accepted" in result:
                if result["accepted"]:
                    self._set_ready(
                        f"Calibration OK (error: {result['mean_error_deg']:.2f} deg)"
                    )
                else:
                    self._set_ready(
                        f"Calibration failed (error: {result['mean_error_deg']:.2f} deg)",
                        success=False,
                    )
            else:
                self._set_ready("Done!")


def run_app() -> None:
    """Launch the OcuTrace GUI application."""
    app = wx.App(False)
    frame = LauncherFrame()
    frame.Show()
    app.MainLoop()


if __name__ == "__main__":
    mp.freeze_support()
    run_app()
