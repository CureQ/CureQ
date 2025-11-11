import os
from pathlib import Path
from typing import Dict, Any, Optional

import ipywidgets as W
from IPython.display import display, clear_output

from ..mea import analyse_wells, get_default_parameters
from ..core._utilities import rechunk_dataset
from ..core._bandpass import butter_bandpass_filter
from ..core._threshold import fast_threshold
from ..core._spike_validation import spike_validation
from ..core._burst_detection import burst_detection
from ..core._network_burst_detection import network_burst_detection
from ..core._plotting import well_electrodes_kde, feature_boxplots, combined_feature_boxplots, features_over_time
from ..core._features import recalculate_features


class CureQApp:
    """
    ipywidgets UI for CureQ/MEAlytics. Designed to run in Jupyter or via Voilà.

    Pages (initial):
    - Analyze file (working)
    - Batch processing (stub)
    - View results (stub)
    - Exclude electrodes (stub)
    - Compress/Rechunk (stub)
    - Parameters (working form, saved to state)
    - Plotting (stub)
    """

    def __init__(self, notebook: bool = True):
        self.notebook = notebook
        self.parameters: Dict[str, Any] = get_default_parameters()
        self.state: Dict[str, Any] = {
            "selected_file": None,
            "sampling_rate": None,
            "electrode_amnt": None,
            "output_folder": None,
        }
        # Build UI
        self._build()

    # ---------- Layout ----------
    def _build(self):
        # Global CSS to aggressively hide horizontal scrollbars inside widgets
        self._global_css = W.HTML("""
        <style>
        /* Hide horizontal scrollbars globally (Notebook + JupyterLab + classic) */
        html, body, .jp-Notebook, .jp-Cell, .jp-OutputArea, .jp-OutputArea-output, .output_wrapper, .output, .prompt, .output_area {
            overflow-x: hidden !important;
        }
        /* Hide horizontal scrollbars for common ipywidgets containers */
        .widget-tab, .widget-box, .widget-vbox, .widget-hbox, .widget-output {
            overflow-x: hidden !important;
            box-sizing: border-box !important;
        }
        /* Also ensure images inside outputs don't cause overflow */
        .widget-output img { max-width: 100% !important; height: auto !important; display: block; }
        </style>
        """)

        self.title = W.HTML("<h2>MEAlytics (ipywidgets)</h2>")

        # Tabs
        self.tabs = W.Tab()
        self.pages = {
            "Analyze file": self._page_analyze_file(),
            "Parameters": self._page_parameters(),
            "Batch processing": self._page_batch_processing(),
            "View results": self._page_view_results(),
            "Exclude electrodes": self._page_exclude_electrodes(),
            "Compress/Rechunk": self._page_compress_rechunk(),
            "Plotting": self._page_plotting(),
        }
        self.tabs.children = list(self.pages.values())
        for i, name in enumerate(self.pages.keys()):
            self.tabs.set_title(i, name)

        # Root
        self.root = W.VBox([self._global_css, self.title, self.tabs])
        # Prevent horizontal scrollbars on the overall UI
        self.root.layout = W.Layout(width='100%', overflow_x='hidden', box_sizing='border-box')
        self.tabs.layout = W.Layout(width='100%', overflow_x='hidden', box_sizing='border-box')
        # Also constrain each page container
        try:
            for child in self.tabs.children:
                child.layout = W.Layout(width='100%', overflow_x='hidden', box_sizing='border-box')
        except Exception:
            pass

    # ---------- Pages ----------
    def _page_stub(self, name: str):
        return W.VBox([
            W.HTML(f"<b>{name}</b>"),
            W.HTML("This page is not implemented yet in the ipywidgets UI."),
        ])

    def _page_analyze_file(self):
        # Controls
        self.file_picker = W.FileUpload(
            accept=".h5",
            multiple=False,
            description="Upload .h5 file",
            style={"button_color": "#4CAF50"},
        )
        self.file_picker.observe(self._on_file_upload, names="value")

        self.file_path_text = W.Text(
            description="File path",
            placeholder="/path/to/experiment.h5 (optional if uploaded)",
            layout=W.Layout(width="100%"),
        )

        self.sampling_rate = W.BoundedIntText(
            description="Sampling rate",
            min=1,
            max=10_000_000,
            step=1,
            value=self.parameters.get("sampling rate", 20000),
        )
        # Make label narrower to avoid horizontal overflow
        try:
            self.sampling_rate.style = {"description_width": "140px"}
        except Exception:
            pass
        self.electrode_amnt = W.BoundedIntText(
            description="Electrodes/well",
            min=1,
            max=4096,
            step=1,
            value=12,
        )
        try:
            self.electrode_amnt.style = {"description_width": "140px"}
        except Exception:
            pass

        self.run_btn = W.Button(
            description="Start analysis",
            button_style="primary",
            icon="play",
        )
        self.run_btn.on_click(self._on_run)

        self.abort_btn = W.Button(
            description="Abort",
            button_style="warning",
            icon="stop",
            disabled=True,
        )
        self.abort_btn.on_click(self._on_abort)

        self.progress = W.FloatProgress(
            description="",
            min=0.0,
            max=1.0,
            value=0.0,
            bar_style="info",
            layout=W.Layout(width="100%"),
        )
        # Optional: tweak bar color via style
        try:
            self.progress.style = {"bar_color": "#3f51b5"}
        except Exception:
            pass
        self.progress_info = W.HTML("")
        # Hide any internal horizontal scrollbar in output
        self.output_area = W.Output(layout=W.Layout(width='100%', overflow_x='hidden'))

        # Helper to create wrapping rows to prevent horizontal scrollbars
        wrap = lambda children: W.Box(children, layout=W.Layout(display='flex', flex_flow='row wrap', width='100%'))

        page = W.VBox([
            W.HTML("<b>Analyze a single MEA experiment</b>"),
            W.HTML("Provide either a local file path or upload a small .h5 file (upload not recommended for very large files)."),
            W.HBox([self.file_path_text], layout=W.Layout(width='100%')),
            wrap([self.file_picker]),
            wrap([self.sampling_rate, self.electrode_amnt]),
            wrap([self.run_btn, self.abort_btn]),
            W.HTML("<b>Progress</b>"),
            self.progress,
            self.progress_info,
            self.output_area,
        ])
        page.layout = W.Layout(width='100%', overflow_x='hidden')
        return page

    def _page_parameters(self):
        # Build simple param form based on default param keys
        widgets = {}
        rows = []
        schema = [
            ("low cutoff", W.BoundedIntText, {"min": 1, "max": 10000, "step": 1}),
            ("high cutoff", W.BoundedIntText, {"min": 10, "max": 50000, "step": 1}),
            ("order", W.BoundedIntText, {"min": 1, "max": 10, "step": 1}),
            ("threshold portion", W.BoundedFloatText, {"min": 0.0, "max": 1.0, "step": 0.01}),
            ("standard deviation multiplier", W.BoundedFloatText, {"min": 0.0, "max": 100.0, "step": 0.1}),
            ("rms multiplier", W.BoundedFloatText, {"min": 0.0, "max": 100.0, "step": 0.1}),
            ("refractory period", W.BoundedFloatText, {"min": 0.0, "max": 1.0, "step": 0.0001}),
            ("spike validation method", W.Dropdown, {"options": ["Noisebased", "none"]}),
            ("exit time", W.BoundedFloatText, {"min": 0.0, "max": 1.0, "step": 0.0001}),
            ("drop amplitude", W.BoundedFloatText, {"min": 0.0, "max": 100.0, "step": 0.1}),
            ("max drop", W.BoundedFloatText, {"min": 0.0, "max": 100.0, "step": 0.1}),
            ("minimal amount of spikes", W.BoundedIntText, {"min": 1, "max": 10000, "step": 1}),
            ("default interval threshold", W.BoundedFloatText, {"min": 0.0, "max": 100000.0, "step": 0.1}),
            ("max interval threshold", W.BoundedFloatText, {"min": 0.0, "max": 100000.0, "step": 0.1}),
            ("burst detection kde bandwidth", W.BoundedFloatText, {"min": 0.0, "max": 100.0, "step": 0.01}),
            ("min channels", W.BoundedFloatText, {"min": 0.0, "max": 1.0, "step": 0.01}),
            ("thresholding method", W.Dropdown, {"options": ["Yen", "Otsu", "Li", "Isodata", "Mean", "Minimum", "Triangle"]}),
            ("nbd kde bandwidth", W.BoundedFloatText, {"min": 0.0, "max": 10.0, "step": 0.001}),
            ("remove inactive electrodes", W.Checkbox, {}),
            ("activity threshold", W.BoundedFloatText, {"min": 0.0, "max": 100.0, "step": 0.01}),
            ("use multiprocessing", W.Checkbox, {}),
        ]
        for key, widget_cls, kwargs in schema:
            val = self.parameters.get(key)
            ctrl = widget_cls(description=key, value=val, **kwargs)
            widgets[key] = ctrl
            rows.append(ctrl)

        self.param_widgets = widgets
        save_btn = W.Button(description="Save parameters", icon="save", button_style="success")
        save_btn.on_click(self._save_params)

        reset_btn = W.Button(description="Reset to defaults", icon="refresh")
        reset_btn.on_click(self._reset_params)

        return W.VBox([
            W.HTML("<b>Analysis parameters</b>"),
            W.VBox(rows, layout=W.Layout(max_height="480px", overflow_y="auto")),
            W.HBox([save_btn, reset_btn])
        ])

    # ---------- Events ----------
    def _on_file_upload(self, change):
        # Save the uploaded file to a temp path the backend can read
        uploads = change["new"]
        if not uploads:
            self.state["selected_file"] = None
            return
        name, meta = next(iter(uploads.items()))
        content = meta["content"]
        # Store in working dir
        tmp_path = Path.cwd() / name
        with open(tmp_path, "wb") as f:
            f.write(content)
        self.state["selected_file"] = str(tmp_path)
        # Also reflect in text field
        self.file_path_text.value = str(tmp_path)

    def _save_params(self, _):
        for k, w in self.param_widgets.items():
            self.parameters[k] = w.value

    def _reset_params(self, _):
        self.parameters = get_default_parameters()
        for k, w in self.param_widgets.items():
            if k in self.parameters:
                try:
                    w.value = self.parameters[k]
                except Exception:
                    pass

    def _on_abort(self, _):
        # Signal the running analyse_wells loop via progress.npy
        sel = self.state.get("selected_file")
        if not sel:
            return
        progressfile = f"{Path(sel).parent}/progress.npy"
        try:
            import numpy as np
            np.save(progressfile, ["abort"])
            self.progress_info.value = "<span style='color:#b71c1c'>Aborting analysis...</span>"
        except Exception:
            pass

    def _on_run(self, _):
        sel = self.file_path_text.value.strip() or self.state.get("selected_file")
        if not sel:
            with self.output_area:
                clear_output()
                print("Please upload a .h5 file.")
            return
        sr = int(self.sampling_rate.value)
        ea = int(self.electrode_amnt.value)

        self.run_btn.disabled = True
        self.abort_btn.disabled = False
        self.progress.value = 0.0
        self.progress_info.value = "Starting..."
        self.output_area.clear_output()

        import threading, time, numpy as np

        def worker():
            try:
                analyse_wells(sel, sampling_rate=sr, electrode_amnt=ea, parameters=self.parameters)
            except Exception as e:
                with self.output_area:
                    print("Error during analysis:", e)
            finally:
                self.abort_btn.disabled = True
                self.run_btn.disabled = False

        def watcher():
            progressfile = f"{Path(sel).parent}/progress.npy"
            start = time.time()
            while True:
                time.sleep(0.2)
                try:
                    arr = np.load(progressfile, allow_pickle=True)
                except Exception:
                    # Not yet created, keep waiting
                    continue
                if isinstance(arr, np.ndarray):
                    arr = arr.tolist()
                if not arr:
                    continue
                status = arr[0]
                if status == "done":
                    self.progress.value = 1.0
                    self.progress_info.value = f"Finished in {time.time()-start:.1f}s"
                    break
                elif status == "rechunking":
                    self.progress_info.value = "Rechunking data..."
                elif status == "starting":
                    self.progress_info.value = "Loading data..."
                elif status == "abort":
                    self.progress_info.value = "Aborting..."
                elif status == "stopped":
                    self.progress_info.value = "Stopped."
                    break
                else:
                    try:
                        cur, total = arr
                        self.progress.value = float(cur) / float(total)
                        self.progress_info.value = f"Analyzing channels: {cur}/{total}"
                    except Exception:
                        pass

        t1 = threading.Thread(target=worker, daemon=True)
        t2 = threading.Thread(target=watcher, daemon=True)
        t1.start(); t2.start()

    # ---------- View results ----------
    def _page_view_results(self):
        self.vr_folder = W.Text(description="Folder", placeholder="/path/to/output_folder", layout=W.Layout(width="100%"))
        self.vr_raw = W.Text(description="Raw .h5", placeholder="/path/to/raw.h5", layout=W.Layout(width="100%"))
        self.vr_load = W.Button(description="Load", icon="folder-open")
        self.vr_load.on_click(self._vr_load)

        self.vr_params: Optional[Dict[str, Any]] = None
        # Selection state
        self.vr_selected_well: Optional[int] = None
        self.vr_selected_electrode: Optional[int] = None

        # Grids placeholders
        self.vr_well_grid_box = W.VBox([])
        self.vr_elec_grid_box = W.VBox([])
        self._vr_well_buttons = []
        self._vr_elec_buttons = []

        # Single-electrode parameter widgets
        def num(label, v, minv, maxv, step):
            return W.BoundedFloatText(description=label, value=float(v), min=float(minv), max=float(maxv), step=float(step), layout=W.Layout(width="260px"))
        def numi(label, v, minv, maxv, step):
            return W.BoundedIntText(description=label, value=int(v), min=int(minv), max=int(maxv), step=int(step), layout=W.Layout(width="260px"))

        # Placeholders; real values set after load
        self.se_low_cut = numi("Low cutoff", 200, 1, 10000, 1)
        self.se_high_cut = numi("High cutoff", 3500, 10, 50000, 1)
        self.se_order = numi("Order", 2, 1, 10, 1)
        self.se_stdev_mult = num("Std dev mult", 5, 0, 100, 0.1)
        self.se_rms_mult = num("RMS mult", 5, 0, 100, 0.1)
        self.se_thresh_portion = num("Thresh portion", 0.1, 0, 1, 0.01)
        self.se_refractory = num("Refractory (s)", 0.001, 0, 1, 0.0001)
        self.se_val_method = W.Dropdown(description="Validation", options=["Noisebased", "none"], value="Noisebased", layout=W.Layout(width="260px"))
        self.se_exit_time = num("Exit time (s)", 0.001, 0, 1, 0.0001)
        self.se_drop_amp = num("Drop amplitude", 5, 0, 100, 0.1)
        self.se_max_drop = num("Max drop", 2, 0, 100, 0.1)
        self.se_plot_rect = W.Checkbox(value=False, description="Plot validation", layout=W.Layout(width="200px"))
        self.vr_se_run = W.Button(description="Update electrode plot", icon="line-chart", button_style="primary")
        self.vr_se_run.on_click(self._vr_show_electrode)

        # Whole-well parameter widgets
        self.ww_min_channels = num("Min channels (0-1)", 0.5, 0, 1, 0.01)
        self.ww_thresh_method = W.Dropdown(description="Thresholding", options=["Yen", "Otsu", "Li", "Isodata", "Mean", "Minimum", "Triangle"], value="Yen", layout=W.Layout(width="260px"))
        self.ww_kde_bw = num("NBD KDE bw", 0.05, 0, 10, 0.001)
        self.ww_elec_kde_bw = num("Well KDE bw", 0.1, 0, 10, 0.001)
        self.vr_ww_run = W.Button(description="Update well plots", icon="bar-chart", button_style="primary")
        self.vr_ww_run.on_click(self._vr_show_well)

        # Plot containers (replace children on each update to avoid duplicates)
        self.vr_elec_box = W.VBox([], layout=W.Layout(width='100%'))
        self.vr_well_box = W.VBox([], layout=W.Layout(width='100%'))

        # Create wrapping rows to avoid horizontal scrollbars
        wrap = lambda children: W.Box(children, layout=W.Layout(display='flex', flex_flow='row wrap', width='100%'))
        page = W.VBox([
            W.HTML("<b>View results</b>"),
            self.vr_folder,
            self.vr_raw,
            self.vr_load,
            W.HTML("<b>Select well</b>"),
            self.vr_well_grid_box,
            W.HTML("<b>Select electrode</b>"),
            self.vr_elec_grid_box,
            W.HTML("<hr><b>Single electrode parameters</b>"),
            wrap([self.se_low_cut, self.se_high_cut, self.se_order]),
            wrap([self.se_stdev_mult, self.se_rms_mult, self.se_thresh_portion]),
            wrap([self.se_refractory, self.se_val_method, self.se_exit_time]),
            wrap([self.se_drop_amp, self.se_max_drop, self.se_plot_rect]),
            self.vr_se_run,
            self.vr_elec_box,
            W.HTML("<hr><b>Whole well parameters</b>"),
            wrap([self.ww_min_channels, self.ww_thresh_method, self.ww_kde_bw, self.ww_elec_kde_bw]),
            self.vr_ww_run,
            self.vr_well_box,
        ])
        page.layout = W.Layout(width='100%', overflow_x='hidden')
        return page

    def _vr_load(self, _):
        import json, h5py
        folder = Path(self.vr_folder.value.strip())
        raw = Path(self.vr_raw.value.strip())
        if not folder.exists() or not (folder / "parameters.json").exists():
            self.vr_elec_box.children = [W.HTML("<span style='color:#b71c1c'>Invalid folder or missing parameters.json</span>")]
            self.vr_well_box.children = []
            return
        if not raw.exists():
            self.vr_elec_box.children = [W.HTML("<span style='color:#b71c1c'>Raw file does not exist</span>")]
            self.vr_well_box.children = []
            return
        try:
            with open(folder / "parameters.json", "r") as f:
                params = json.load(f)
        except Exception as e:
            self.vr_elec_box.children = [W.HTML(f"<span style='color:#b71c1c'>Could not load parameters: {e}</span>")]
            self.vr_well_box.children = []
            return
        self.vr_params = params
        # Infer wells and electrodes
        try:
            with h5py.File(str(raw), "r") as hf:
                shape = hf["Data/Recording_0/AnalogStream/Stream_0/ChannelData"].shape
        except Exception as e:
            self.vr_elec_box.children = [W.HTML(f"<span style='color:#b71c1c'>Could not open raw file: {e}</span>")]
            self.vr_well_box.children = []
            return
        electrode_amnt = int(params.get("electrode amount", 12))
        # Determine which axis corresponds to channels to avoid gigantic grids
        channels_dim = self._vr_find_channels_dim(shape, electrode_amnt)
        channels = int(shape[channels_dim])
        well_amnt = int(channels // electrode_amnt)
        if well_amnt <= 0 or well_amnt > 1536:
            self.vr_elec_box.children = [W.HTML(
                f"<span style='color:#b71c1c'>Suspicious dataset shape {shape} with electrode_amnt={electrode_amnt}.<br/>"
                f"Detected wells={well_amnt}. Please verify electrode amount or dataset orientation.</span>"
            )]
            self.vr_well_box.children = []
            return
        # Populate parameter widgets with loaded defaults
        self._vr_apply_param_defaults(params)
        # Build selection grids
        self._vr_build_well_grid(well_amnt)
        self._vr_build_electrode_grid(electrode_amnt)
        # Default selection
        self.vr_selected_well = 1 if well_amnt > 0 else None
        self.vr_selected_electrode = 1 if electrode_amnt > 0 else None
        self._vr_refresh_selection_styles()
        # Save paths and channel axis
        self.state["vr_folder"] = str(folder)
        self.state["vr_raw"] = str(raw)
        self.state["vr_channels_dim"] = int(channels_dim)

    def _vr_show_electrode(self, _):
        if not self.vr_params:
            return
        folder = Path(self.state.get("vr_folder", ""))
        raw = Path(self.state.get("vr_raw", ""))
        if not folder or not raw:
            return
        if not self.vr_selected_well or not self.vr_selected_electrode:
            return
        well = int(self.vr_selected_well)
        electrode_idx = int(self.vr_selected_electrode)
        params = dict(self.vr_params)
        params["output hdf file"] = str(folder / "output_values.h5")
        params["output path"] = str(folder)
        # Override with UI entries
        params['low cutoff'] = int(self.se_low_cut.value)
        params['high cutoff'] = int(self.se_high_cut.value)
        params['order'] = int(self.se_order.value)
        params['standard deviation multiplier'] = float(self.se_stdev_mult.value)
        params['rms multiplier'] = float(self.se_rms_mult.value)
        params['threshold portion'] = float(self.se_thresh_portion.value)
        params['refractory period'] = float(self.se_refractory.value)
        params['spike validation method'] = str(self.se_val_method.value)
        if params['spike validation method'] == 'none':
            params['drop amplitude'] = 0
        else:
            params['exit time'] = float(self.se_exit_time.value)
            params['drop amplitude'] = float(self.se_drop_amp.value)
            params['max drop'] = float(self.se_max_drop.value)

        # Map well/electrode to channel index
        e_amnt = int(params["electrode amount"]) 
        channel = (well - 1) * e_amnt + electrode_idx - 1

        import h5py
        try:
            with h5py.File(str(raw), "r") as hf:
                ds = hf["Data/Recording_0/AnalogStream/Stream_0/ChannelData"]
                ch_dim = int(self.state.get("vr_channels_dim", 0))
                if ch_dim == 0:
                    data = ds[channel]
                elif ch_dim == 1:
                    data = ds[:, channel]
                else:
                    # generic slicing for N-D (unlikely)
                    slc = [slice(None)] * len(ds.shape)
                    slc[ch_dim] = channel
                    data = ds[tuple(slc)]
        except Exception as e:
            with self.vr_out_electrode:
                print("Could not read data:", e)
            return

        import matplotlib.pyplot as plt
        # Suppress any internal display calls during figure creation
        with self._suppress_mpl_display():
            filt = butter_bandpass_filter(data, params)
            thr = fast_threshold(filt, params)
            fig_spk = spike_validation(
                filt, channel, thr, params,
                plot_electrodes=True, savedata=False,
                plot_rectangles=bool(self.se_plot_rect.value)
            )
            _, fig_burst = burst_detection(
                filt, channel, params, plot_electrodes=True, savedata=False
            )

        # Render figures as PNG images by replacing the container children
        w1 = self._fig_to_img_widget(fig_spk, dpi=140)
        w2 = self._fig_to_img_widget(fig_burst, dpi=140)
        self.vr_elec_box.children = [w1, w2]
        plt.close(fig_spk)
        plt.close(fig_burst)

    def _vr_show_well(self, _):
        if not self.vr_params:
            return
        if not self.vr_selected_well:
            return
        folder = Path(self.state.get("vr_folder", ""))
        params = dict(self.vr_params)
        params["output hdf file"] = str(folder / "output_values.h5")
        params["output path"] = str(folder)
        well = int(self.vr_selected_well)
        # Override NBD params
        params["min channels"] = float(self.ww_min_channels.value)
        params["thresholding method"] = str(self.ww_thresh_method.value)
        params["nbd kde bandwidth"] = float(self.ww_kde_bw.value)

        import matplotlib.pyplot as plt
        # Suppress internal displays during figure creation
        with self._suppress_mpl_display():
            fig_nbd = network_burst_detection([well], params, plot_electrodes=True, savedata=False, save_figures=False)
            fig_kde = well_electrodes_kde(str(folder), well, params, bandwidth=float(self.ww_elec_kde_bw.value))

        w1 = self._fig_to_img_widget(fig_nbd, dpi=140)
        w2 = self._fig_to_img_widget(fig_kde, dpi=140)
        self.vr_well_box.children = [w1, w2]
        plt.close(fig_nbd)
        plt.close(fig_kde)

    # Utility: suppress matplotlib/IPython displays inside callbacks to avoid duplicate plots
    from contextlib import contextmanager
    @contextmanager
    def _suppress_mpl_display(self):
        try:
            import matplotlib.pyplot as plt
        except Exception:
            plt = None
        try:
            from IPython import display as IPdisplay
        except Exception:
            IPdisplay = None
        old_show = getattr(plt, 'show', None) if plt else None
        old_display = getattr(IPdisplay, 'display', None) if IPdisplay else None
        def _noop(*args, **kwargs):
            return None
        try:
            if plt and old_show:
                plt.show = _noop
            if IPdisplay and old_display:
                IPdisplay.display = _noop
            yield
        finally:
            if plt and old_show:
                plt.show = old_show
            if IPdisplay and old_display:
                IPdisplay.display = old_display

    # ---------- View results helpers ----------
    def _vr_find_channels_dim(self, shape, electrode_amnt: int) -> int:
        # Try to locate axis divisible by electrode count with reasonable well count
        try:
            dims = tuple(int(x) for x in shape)
        except Exception:
            return 0
        candidates = []
        for i, s in enumerate(dims):
            if s % max(1, int(electrode_amnt)) == 0:
                wells = s // max(1, int(electrode_amnt))
                if 1 <= wells <= 384:  # typical upper bound
                    candidates.append((wells, i))
        if candidates:
            # Prefer the smallest reasonable wells count
            candidates.sort()
            return candidates[0][1]
        # Fallback: choose the smaller dimension if 2D, else 0
        if len(dims) == 2:
            return 0 if dims[0] <= dims[1] else 1
        return 0
    def _vr_apply_param_defaults(self, params: Dict[str, Any]):
        # Populate UI from parameters.json defaults
        self.se_low_cut.value = int(params.get('low cutoff', 200))
        self.se_high_cut.value = int(params.get('high cutoff', 3500))
        self.se_order.value = int(params.get('order', 2))
        self.se_stdev_mult.value = float(params.get('standard deviation multiplier', 5))
        self.se_rms_mult.value = float(params.get('rms multiplier', 5))
        self.se_thresh_portion.value = float(params.get('threshold portion', 0.1))
        self.se_refractory.value = float(params.get('refractory period', 0.001))
        self.se_val_method.value = str(params.get('spike validation method', 'Noisebased'))
        self.se_exit_time.value = float(params.get('exit time', 0.001))
        self.se_drop_amp.value = float(params.get('drop amplitude', 5))
        self.se_max_drop.value = float(params.get('max drop', 2))
        self.ww_min_channels.value = float(params.get('min channels', 0.5))
        self.ww_thresh_method.value = str(params.get('thresholding method', 'Yen'))
        self.ww_kde_bw.value = float(params.get('nbd kde bandwidth', 0.05))
        self.ww_elec_kde_bw.value = 0.1

    def _vr_build_well_grid(self, well_amnt: int):
        # Compute grid similar to original GUI: try to make near-square, with 24 preset (6x4)
        # For large well counts, use a compact Dropdown to avoid building thousands of buttons.
        if well_amnt > 96:
            dd = W.Dropdown(description="Well", options=list(range(1, well_amnt+1)), value=self.vr_selected_well or 1, layout=W.Layout(width="260px"))
            def on_change(change):
                if change.get('name') == 'value':
                    self.vr_selected_well = int(change['new'])
                    self._vr_refresh_selection_styles()
            dd.observe(on_change, names='value')
            self._vr_well_buttons = []
            self.vr_well_grid_box.children = [dd]
            return
        def calc_well_grid(n: int):
            if n == 24:
                return 6, 4
            import math
            best = (n, 1)
            min_diff = n
            for w in range(1, int(math.sqrt(n)) + 1):
                if n % w == 0:
                    h = n // w
                    if abs(w - h) < min_diff:
                        min_diff = abs(w - h)
                        best = (max(w, h), min(w, h))
            return best
        xw, yw = calc_well_grid(well_amnt)
        buttons = []
        rows = []
        num = 1
        def make_cb(idx):
            def _cb(_):
                self.vr_selected_well = idx
                self._vr_refresh_selection_styles()
            return _cb
        for y in range(yw):
            row_btns = []
            for x in range(xw):
                if num <= well_amnt:
                    b = W.Button(description=str(num), layout=W.Layout(width="48px"))
                    b.on_click(make_cb(num))
                    row_btns.append(b)
                    buttons.append(b)
                    num += 1
            rows.append(W.HBox(row_btns))
        self._vr_well_buttons = buttons
        self.vr_well_grid_box.children = rows

    def _vr_build_electrode_grid(self, electrode_amnt: int):
        # Build layout for 12 or 16 specifically; otherwise square grid
        import numpy as np
        if electrode_amnt == 12:
            layout = np.array([[0,1,1,0],[1,1,1,1],[1,1,1,1],[0,1,1,0]], dtype=bool)
        elif electrode_amnt == 16:
            layout = np.ones((4,4), dtype=bool)
        else:
            # square-ish grid
            side = int(np.ceil(np.sqrt(electrode_amnt)))
            layout = np.zeros((side, side), dtype=bool)
            count = 0
            for i in range(side):
                for j in range(side):
                    if count < electrode_amnt:
                        layout[i,j] = True
                        count += 1
        buttons = []
        rows = []
        num = 1
        def make_cb(idx):
            def _cb(_):
                self.vr_selected_electrode = idx
                self._vr_refresh_selection_styles()
            return _cb
        for i in range(layout.shape[0]):
            row_btns = []
            for j in range(layout.shape[1]):
                if layout[i,j]:
                    b = W.Button(description=str(num), layout=W.Layout(width="48px"))
                    b.on_click(make_cb(num))
                    row_btns.append(b)
                    buttons.append(b)
                    num += 1
                else:
                    # spacer
                    row_btns.append(W.HTML("&nbsp;&nbsp;&nbsp;&nbsp;"))
            rows.append(W.HBox(row_btns))
        self._vr_elec_buttons = buttons
        self.vr_elec_grid_box.children = rows

    def _vr_refresh_selection_styles(self):
        # Mark selected well/electrode buttons
        for i, b in enumerate(getattr(self, '_vr_well_buttons', []), start=1):
            b.button_style = 'primary' if self.vr_selected_well == i else ''
        for i, b in enumerate(getattr(self, '_vr_elec_buttons', []), start=1):
            b.button_style = 'primary' if self.vr_selected_electrode == i else ''

    def _display_fig(self, fig, out_widget, dpi=120):
        # Convert a Matplotlib Figure to PNG and display as a responsive <img> (no scrollbars)
        from io import BytesIO
        from IPython.display import HTML
        import base64
        buf = BytesIO()
        try:
            fig.savefig(buf, format='png', dpi=int(dpi), bbox_inches='tight')
            data = buf.getvalue()
            b64 = base64.b64encode(data).decode('ascii')
            html = f"<img src='data:image/png;base64,{b64}' style='max-width:100%; height:auto; display:block;'/>"
            with out_widget:
                display(HTML(html))
        finally:
            buf.close()

    def _fig_to_img_widget(self, fig, dpi=120):
        # Convert a Matplotlib Figure to a standalone HTML widget with an <img> tag
        from io import BytesIO
        import base64
        buf = BytesIO()
        try:
            fig.savefig(buf, format='png', dpi=int(dpi), bbox_inches='tight')
            data = buf.getvalue()
            b64 = base64.b64encode(data).decode('ascii')
            return W.HTML(f"<img src='data:image/png;base64,{b64}' style='max-width:100%; height:auto; display:block;' />")
        finally:
            buf.close()

    # ---------- Batch processing ----------
    def _page_batch_processing(self):
        self.batch_folder = W.Text(description="Folder", placeholder="/path/to/folder", layout=W.Layout(width="100%"))
        self.batch_scan_btn = W.Button(description="Scan", icon="search")
        self.batch_scan_btn.on_click(self._batch_scan)

        self.batch_files = W.SelectMultiple(options=[], rows=10, description="Files")

        self.batch_sampling_rate = W.BoundedIntText(description="Sampling rate", min=1, max=10_000_000, value=20000)
        self.batch_electrode_amnt = W.BoundedIntText(description="Electrodes/well", min=1, max=4096, value=12)

        self.batch_run_btn = W.Button(description="Run batch", icon="play", button_style="primary")
        self.batch_run_btn.on_click(self._batch_run)
        self.batch_abort_btn = W.Button(description="Abort after current", icon="stop", button_style="warning", disabled=True)
        self.batch_abort_btn.on_click(self._batch_abort)
        self.batch_abort_flag = False

        self.batch_overall = W.FloatProgress(description="Overall", min=0.0, max=1.0, value=0.0, layout=W.Layout(width="100%"))
        self.batch_status = W.HTML("")
        self.batch_per_file_box = W.VBox([])

        return W.VBox([
            W.HTML("<b>Batch processing</b>"),
            W.HBox([self.batch_folder, self.batch_scan_btn]),
            self.batch_files,
            W.HBox([self.batch_sampling_rate, self.batch_electrode_amnt]),
            W.HBox([self.batch_run_btn, self.batch_abort_btn]),
            self.batch_overall,
            self.batch_status,
            W.HTML("Per-file progress"),
            self.batch_per_file_box,
        ])

    def _batch_scan(self, _):
        folder = Path(self.batch_folder.value.strip())
        if not folder.exists() or not folder.is_dir():
            self.batch_status.value = "<span style='color:#b71c1c'>Invalid folder</span>"
            return
        files = []
        for root, _, fnames in os.walk(folder):
            for n in fnames:
                if n.endswith(".h5"):
                    files.append(str(Path(root) / n))
        files.sort()
        self.batch_files.options = files
        self.batch_status.value = f"Found {len(files)} .h5 files"

    def _batch_abort(self, _):
        self.batch_abort_flag = True
        self.batch_abort_btn.disabled = True
        self.batch_status.value = "Aborting after current file..."

    def _batch_run(self, _):
        selected = list(self.batch_files.value)
        if not selected:
            self.batch_status.value = "<span style='color:#b71c1c'>No files selected</span>"
            return
        sr = int(self.batch_sampling_rate.value)
        ea = int(self.batch_electrode_amnt.value)

        # UI setup
        self.batch_run_btn.disabled = True
        self.batch_abort_btn.disabled = False
        self.batch_overall.value = 0.0
        self.batch_status.value = "Starting batch..."
        self.batch_per_file_box.children = []

        # Create progress widgets per file
        per_file = []
        for f in selected:
            bar = W.FloatProgress(min=0.0, max=1.0, value=0.0, layout=W.Layout(width="100%"))
            per_file.append((f, bar))
        self.batch_per_file_box.children = [
            W.VBox([W.HTML(Path(f).name), bar]) for f, bar in per_file
        ]

        import threading, time, numpy as np

        def run_seq():
            crashed = False
            for idx, (f, bar) in enumerate(per_file):
                if self.batch_abort_flag:
                    self.batch_status.value = "Aborted by user."
                    break

                def worker():
                    nonlocal crashed
                    try:
                        analyse_wells(f, sampling_rate=sr, electrode_amnt=ea, parameters=self.parameters)
                    except Exception as e:
                        crashed = True
                        bar.bar_style = "danger"
                        with self.output_area:
                            print("Error in", f, e)

                def watcher():
                    progressfile = f"{Path(f).parent}/progress.npy"
                    # remove pre-existing
                    try:
                        os.remove(progressfile)
                    except Exception:
                        pass
                    while True:
                        if crashed:
                            break
                        try:
                            arr = np.load(progressfile, allow_pickle=True)
                        except Exception:
                            time.sleep(0.2)
                            continue
                        if isinstance(arr, np.ndarray):
                            arr = arr.tolist()
                        status = arr[0]
                        if status == "done":
                            bar.value = 1.0
                            break
                        elif status == "stopped":
                            break
                        elif status == "rechunking":
                            self.batch_status.value = f"Rechunking: {Path(f).name}"
                        elif status == "starting":
                            self.batch_status.value = f"Loading: {Path(f).name}"
                        else:
                            try:
                                cur, total = arr
                                bar.value = float(cur) / float(total)
                            except Exception:
                                pass
                        time.sleep(0.2)

                t1 = threading.Thread(target=worker, daemon=True)
                t2 = threading.Thread(target=watcher, daemon=True)
                t1.start(); t2.start()
                t1.join(); t2.join()
                self.batch_overall.value = (idx + 1) / len(per_file)

            self.batch_abort_btn.disabled = True
            self.batch_run_btn.disabled = False
            self.batch_abort_flag = False
            self.batch_status.value = "Batch finished."

        threading.Thread(target=run_seq, daemon=True).start()

    # ---------- Compress/Rechunk ----------
    def _page_compress_rechunk(self):
        self.comp_file = W.Text(description="File", placeholder="/path/to/file.h5", layout=W.Layout(width="100%"))
        self.comp_all = W.Checkbox(value=False, description="Compress all .h5 in folder")
        self.comp_method = W.Dropdown(options=["lzf", "gzip"], value="lzf", description="Method")
        self.comp_level = W.IntSlider(value=1, min=1, max=9, step=1, description="GZIP level")
        def _toggle_level(change=None):
            self.comp_level.disabled = self.comp_method.value != "gzip"
        self.comp_method.observe(_toggle_level, names="value")
        _toggle_level()

        self.comp_run = W.Button(description="Start", icon="cog", button_style="primary")
        self.comp_run.on_click(self._compress_run)
        self.comp_status = W.HTML("")
        self.comp_output = W.Output()

        return W.VBox([
            W.HTML("<b>Compress/Rechunk files</b>"),
            self.comp_file,
            self.comp_all,
            W.HBox([self.comp_method, self.comp_level]),
            self.comp_run,
            self.comp_status,
            self.comp_output,
        ])

    def _compress_run(self, _):
        path = self.comp_file.value.strip()
        if not path:
            self.comp_status.value = "<span style='color:#b71c1c'>Provide a file path</span>"
            return
        p = Path(path)
        if not p.exists():
            self.comp_status.value = "<span style='color:#b71c1c'>Path does not exist</span>"
            return
        files = []
        if self.comp_all.value and p.is_file():
            files = [str(Path(p.parent) / f) for f in os.listdir(p.parent) if f.endswith('.h5')]
        elif p.is_dir():
            files = [str(Path(p) / f) for f in os.listdir(p) if f.endswith('.h5')]
        else:
            files = [str(p)]

        method = self.comp_method.value
        level = int(self.comp_level.value)

        self.comp_status.value = f"Processing {len(files)} file(s)..."
        self.comp_run.disabled = True

        import threading

        def worker():
            ok, fail = [], []
            for f in files:
                try:
                    rechunk_dataset(fileadress=f, compression_method=method, compression_level=level, always_compress_files=True)
                    ok.append(f)
                except Exception as e:
                    fail.append((f, e))
            with self.comp_output:
                from pprint import pprint
                print("Finished compression.")
                if ok:
                    print("Compressed:")
                    pprint(ok)
                if fail:
                    print("Failed:")
                    for f, e in fail:
                        print(" -", f, "=>", e)
            self.comp_run.disabled = False
            self.comp_status.value = "Done."

        threading.Thread(target=worker, daemon=True).start()

    # ---------- Plotting ----------
    def _page_plotting(self):
        self.plt_folder = W.Text(description="Folder", placeholder="/path/to/results/folder", layout=W.Layout(width="100%"))
        self.plt_mode = W.Dropdown(options=[
            "Combine feature boxplots",
            "Features over time",
        ], description="Mode")
        self.plt_labels = W.Textarea(description="Labels JSON", placeholder='{"GroupA": [1,2,3], "GroupB": [4,5,6]}', layout=W.Layout(width="100%", height="120px"))
        self.plt_output = W.Text(description="Output PDF", placeholder="/path/to/output.pdf", layout=W.Layout(width="100%"))
        self.plt_colors = W.Text(description="Colors (opt)", placeholder="#ff0000,#00ff00,#0000ff", layout=W.Layout(width="100%"))
        self.plt_show_pts = W.Checkbox(value=True, description="Show datapoints")
        self.plt_discern = W.Checkbox(value=False, description="Discern wells (colors)")
        self.plt_div_prefix = W.Text(description="DIV prefix", placeholder="DIV")
        self.plt_well_amnt = W.BoundedIntText(description="Well count", min=1, max=1536, value=24)
        self.plt_run = W.Button(description="Generate PDF", icon="file-pdf-o", button_style="primary")
        self.plt_run.on_click(self._plotting_run)
        self.plt_status = W.HTML("")

        def _toggle_fields(change=None):
            is_time = self.plt_mode.value == "Features over time"
            self.plt_div_prefix.disabled = not is_time
            self.plt_well_amnt.disabled = is_time
            self.plt_discern.disabled = is_time
        self.plt_mode.observe(_toggle_fields, names="value")
        _toggle_fields()

        return W.VBox([
            W.HTML("<b>Plotting</b>"),
            self.plt_folder,
            self.plt_mode,
            self.plt_labels,
            self.plt_output,
            self.plt_colors,
            W.HBox([self.plt_show_pts, self.plt_discern]),
            W.HBox([self.plt_div_prefix, self.plt_well_amnt]),
            self.plt_run,
            self.plt_status,
        ])

    def _plotting_run(self, _):
        import json
        folder = self.plt_folder.value.strip()
        output = self.plt_output.value.strip() or str(Path(folder) / "plots.pdf")
        try:
            labels = json.loads(self.plt_labels.value or "{}")
        except Exception as e:
            self.plt_status.value = f"<span style='color:#b71c1c'>Invalid labels JSON: {e}</span>"
            return
        colors = [c.strip() for c in self.plt_colors.value.split(',') if c.strip()] or None
        try:
            if self.plt_mode.value == "Combine feature boxplots":
                pdf = combined_feature_boxplots(
                    folder=folder,
                    labels=labels,
                    output_fileadress=output,
                    colors=colors,
                    show_datapoints=bool(self.plt_show_pts.value),
                    discern_wells=bool(self.plt_discern.value),
                    well_amnt=int(self.plt_well_amnt.value),
                )
            else:
                pdf = features_over_time(
                    folder=folder,
                    labels=labels,
                    div_prefix=self.plt_div_prefix.value or "DIV",
                    output_fileadress=output,
                    colors=colors,
                    show_datapoints=bool(self.plt_show_pts.value),
                )
            self.plt_status.value = f"Generated: {pdf}"
        except Exception as e:
            self.plt_status.value = f"<span style='color:#b71c1c'>Error: {e}</span>"

    # ---------- Exclude Electrodes ----------
    def _page_exclude_electrodes(self):
        self.ee_folder = W.Text(description="Folder", placeholder="/path/to/results/root", layout=W.Layout(width="100%"))
        self.ee_scan = W.Button(description="Scan", icon="search")
        self.ee_scan.on_click(self._ee_scan)
        self.ee_files = W.SelectMultiple(options=[], rows=8, description="Files")
        self.ee_status = W.HTML("")

        self.ee_wells = 0
        self.ee_elec = 0
        self.ee_grid_box = W.VBox([])
        self.ee_toggles = []  # list[widgets.ToggleButton]

        self.ee_save = W.Button(description="Save config", icon="save")
        self.ee_save.on_click(self._ee_save)
        self.ee_load = W.Button(description="Load config", icon="folder-open")
        self.ee_load.on_click(self._ee_load)
        self.ee_run = W.Button(description="Recalculate features", icon="cogs", button_style="primary")
        self.ee_run.on_click(self._ee_run)
        self.ee_progress = W.FloatProgress(min=0.0, max=1.0, value=0.0, layout=W.Layout(width="100%"))
        self.ee_out = W.Output()

        return W.VBox([
            W.HTML("<b>Exclude electrodes and recalculate features</b>"),
            W.HBox([self.ee_folder, self.ee_scan]),
            self.ee_files,
            self.ee_status,
            W.HTML("Electrode configuration (per-well)"),
            self.ee_grid_box,
            W.HBox([self.ee_save, self.ee_load, self.ee_run]),
            self.ee_progress,
            self.ee_out,
        ])

    def _ee_scan(self, _):
        folder = Path(self.ee_folder.value.strip())
        if not folder.exists():
            self.ee_status.value = "<span style='color:#b71c1c'>Invalid folder</span>"
            return
        feature_files = []
        wells_set = []
        elecs_set = []
        for root, _, files in os.walk(folder):
            for n in files:
                if n.endswith("Features.csv") and "Electrode" not in n:
                    fpath = Path(root) / n
                    feature_files.append(str(fpath))
                    try:
                        import pandas as pd, json
                        df = pd.read_csv(str(fpath))
                        wells_set.append(len(df))
                        with open(Path(root) / "parameters.json", "r") as jf:
                            params = json.load(jf)
                        elecs_set.append(int(params["electrode amount"]))
                    except Exception:
                        pass
        feature_files.sort()
        self.ee_files.options = feature_files
        if feature_files and wells_set and elecs_set and min(wells_set) == max(wells_set) and min(elecs_set) == max(elecs_set):
            self.ee_wells = int(wells_set[0])
            self.ee_elec = int(elecs_set[0])
            self._ee_build_grid()
            self.ee_status.value = f"Found {len(feature_files)} files. Wells={self.ee_wells}, Electrodes/well={self.ee_elec}"
        else:
            self.ee_status.value = "<span style='color:#b71c1c'>Files differ in layout or none found</span>"

    def _ee_build_grid(self):
        # Create per-well grids using physical-like layouts (12 diamond / 16 square).
        import numpy as np
        self.ee_toggles = []
        well_boxes = []

        def make_layout(n: int):
            if n == 12:
                return np.array([[0,1,1,0],
                                 [1,1,1,1],
                                 [1,1,1,1],
                                 [0,1,1,0]], dtype=bool)
            if n == 16:
                return np.ones((4,4), dtype=bool)
            # square-ish fill for other counts
            side = int(np.ceil(np.sqrt(n)))
            layout = np.zeros((side, side), dtype=bool)
            count = 0
            for i in range(side):
                for j in range(side):
                    if count < n:
                        layout[i, j] = True
                        count += 1
            return layout

        base_layout = make_layout(self.ee_elec)

        for w in range(self.ee_wells):
            idx = 1
            rows = []
            for i in range(base_layout.shape[0]):
                row_widgets = []
                for j in range(base_layout.shape[1]):
                    if base_layout[i, j] and idx <= self.ee_elec:
                        t = W.ToggleButton(description=str(idx), value=True, layout=W.Layout(width="40px"))
                        row_widgets.append(t)
                        self.ee_toggles.append(t)
                        idx += 1
                    else:
                        row_widgets.append(W.HTML("&nbsp;&nbsp;&nbsp;&nbsp;"))
                rows.append(W.HBox(row_widgets))
            grid = W.VBox(rows)
            well_box = W.VBox([W.HTML(f"<b>Well {w+1}</b>"), grid])
            well_boxes.append(well_box)

        self.ee_grid_box.children = [
            W.VBox(well_boxes, layout=W.Layout(max_height="400px", overflow_y="auto"))
        ]

    def _ee_save(self, _):
        import numpy as np
        if not self.ee_toggles:
            return
        arr = np.array([bool(t.value) for t in self.ee_toggles], dtype=bool)
        path = str(Path(self.ee_folder.value.strip()) / "electrode_config.npy")
        try:
            np.save(path, arr)
            self.ee_status.value = f"Saved: {path}"
        except Exception as e:
            self.ee_status.value = f"<span style='color:#b71c1c'>Save failed: {e}</span>"

    def _ee_load(self, _):
        import numpy as np
        path = str(Path(self.ee_folder.value.strip()) / "electrode_config.npy")
        try:
            arr = np.load(path)
        except Exception as e:
            self.ee_status.value = f"<span style='color:#b71c1c'>Load failed: {e}</span>"
            return
        if len(arr) != len(self.ee_toggles):
            self.ee_status.value = "<span style='color:#b71c1c'>Config shape mismatch</span>"
            return
        for v, t in zip(arr.tolist(), self.ee_toggles):
            t.value = bool(v)
        self.ee_status.value = f"Loaded: {path}"

    def _ee_run(self, _):
        import numpy as np, json, pandas as pd
        selected = list(self.ee_files.value)
        if not selected:
            self.ee_status.value = "<span style='color:#b71c1c'>No files selected</span>"
            return
        cfg = np.array([bool(t.value) for t in self.ee_toggles], dtype=bool).tolist()

        self.ee_progress.value = 0.0
        self.ee_run.disabled = True

        import threading

        def worker():
            ok, fail = [], []
            for i, f in enumerate(selected):
                folder = str(Path(f).parent)
                try:
                    with open(Path(folder) / "parameters.json", "r") as jf:
                        params = json.load(jf)
                    recalculate_features(outputfolder=folder, well_amnt=self.ee_wells, electrode_amnt=self.ee_elec, electrodes=cfg, sampling_rate=params["sampling rate"], measurements=params["measurements"])
                    ok.append(f)
                except Exception as e:
                    fail.append((f, e))
                self.ee_progress.value = (i+1) / len(selected)
            self.ee_run.disabled = False
            with self.ee_out:
                from pprint import pprint
                print("Recalculation finished.")
                if ok:
                    print("Updated:")
                    pprint(ok)
                if fail:
                    print("Failed:")
                    for f, e in fail:
                        print(" -", f, "=>", e)
        threading.Thread(target=worker, daemon=True).start()

    # ---------- Public API ----------
    def widget(self):
        return self.root

    def show(self):
        display(self.root)
