import seaborn as sns
import pandas as pd
from typing import Any, Callable, Optional
from utils.parameter_handling import load_parameters
from utils.log_handling import log_error, log_info, log_warn, log_dict
import matplotlib.pyplot as plt
import os


class Plotter:
    """
    A class for handling plot styling and size parameters.

    Principle: All plotting logic should be done elsewhere and captured in plot_functions. This class should just handle the styling and showing/saving of the plots.

    Typical Workflows:
    1. Create a Plotter object
    2. Create a plot function using get_stacked_bar_plot_func or by defining your own function that creates a plot.
    3. Use test_sizes to test different sizes for the plot and find the best one. This will show the plot and ask if you want to keep trying different sizes.
    4. Do not use plt.show(), instead use plotter.show()
    """

    COLOURS = (
        []
    )  # Add your desired colours here, maybe even change this to a dict if you want

    def __init__(self, parameters: Optional[dict[str, Any]] = None) -> None:
        """
        Initialise the Plotter, load parameters, and apply default seaborn styling.

        :param parameters: Loaded parameters dict. If None, parameters are loaded from disk.
        :type parameters: dict[str, Any] or None
        """
        self.parameters = load_parameters(parameters)
        self.size_params = {}
        sns.set_style("whitegrid")
        self.default_plt_params = plt.rcParams.copy()
        self.set_size_parameters()

    def set_size_parameters(
        self,
        scaler: float = 1,
        font_size: Optional[float] = 16,
        labels_font_size: Optional[float] = 19,
        xtick_font_size: Optional[float] = 19,
        ytick_font_size: Optional[float] = 15,
        legend_font_size: Optional[float] = 16,
        title_font_size: Optional[float] = 20,
    ) -> None:
        """
        Set matplotlib font size parameters, applying an optional uniform scaler.

        Passing ``None`` for any size parameter falls back to the matplotlib default
        for that setting. All values are stored in ``self.size_params`` after scaling.

        :param scaler: Multiplicative scaler applied to all font sizes.
        :type scaler: float
        :param font_size: Base font size. If None, uses the matplotlib default.
        :type font_size: float or None
        :param labels_font_size: Axis label font size. If None, uses the matplotlib default.
        :type labels_font_size: float or None
        :param xtick_font_size: X-axis tick label font size. If None, uses the matplotlib default.
        :type xtick_font_size: float or None
        :param ytick_font_size: Y-axis tick label font size. If None, uses the matplotlib default.
        :type ytick_font_size: float or None
        :param legend_font_size: Legend font size. If None, uses the matplotlib default.
        :type legend_font_size: float or None
        :param title_font_size: Axes title font size. If None, uses the matplotlib default.
        :type title_font_size: float or None
        """
        if font_size is None:
            font_size = self.default_plt_params["font.size"]
        plt.rcParams.update({"font.size": font_size * scaler})
        if labels_font_size is None:
            labels_font_size = self.default_plt_params["axes.labelsize"]
        plt.rcParams.update({"axes.labelsize": labels_font_size * scaler})
        if xtick_font_size is None:
            xtick_font_size = self.default_plt_params["xtick.labelsize"]
        plt.rcParams.update({"xtick.labelsize": xtick_font_size * scaler})
        if ytick_font_size is None:
            ytick_font_size = self.default_plt_params["ytick.labelsize"]
        plt.rcParams.update({"ytick.labelsize": ytick_font_size * scaler})
        if title_font_size is None:
            title_font_size = self.default_plt_params["axes.titlesize"]
        plt.rcParams.update({"axes.titlesize": title_font_size * scaler})
        self.size_params["font_size"] = font_size * scaler
        self.size_params["labels_font_size"] = labels_font_size * scaler
        self.size_params["xtick_font_size"] = xtick_font_size * scaler
        self.size_params["ytick_font_size"] = ytick_font_size * scaler
        self.size_params["title_font_size"] = title_font_size * scaler
        self.size_params["legend_font_size"] = legend_font_size * scaler
        return

    def set_size_default(self, scaler: float = 1) -> None:
        """
        Reset all font size parameters to matplotlib defaults, with an optional scaler.

        :param scaler: Multiplicative scaler applied to all default font sizes.
        :type scaler: float
        """
        self.set_size_parameters(
            scaler=scaler,
            font_size=None,
            labels_font_size=None,
            xtick_font_size=None,
            ytick_font_size=None,
            legend_font_size=None,
            title_font_size=None,
        )
        return

    def set_size_parameters_from_dict(self, size_params: dict[str, float]) -> None:
        """
        Set size parameters from a dictionary. This trusts that the dictionary is correct and does not check for errors.

        :param size_params: Dictionary of size parameter names to values, matching
            the keyword arguments of ``set_size_parameters``.
        :type size_params: dict[str, float]
        """
        self.set_size_parameters(**size_params)
        return

    def get_size_input_number(self, key_name: str) -> float:
        """
        Interactively prompt the user to enter a new value for a size parameter.

        Loops until a valid positive float is provided. Pressing enter with no
        input keeps the current value.

        :param key_name: The name of the size parameter to update (must be a key in ``self.size_params``).
        :type key_name: str
        :return: The new (or unchanged) float value for the parameter.
        :rtype: float
        """
        while True:
            got = input(
                f"Enter the size for {key_name} (current value is {self.size_params[key_name]}, hit enter to keep current value ):"
            )
            got = got.strip()
            if got.strip() == "":
                return self.size_params[key_name]
            try:
                got = float(got)
                if got <= 0:
                    log_warn(
                        f"Got {got}, but it must be greater than 0",
                        parameters=self.parameters,
                    )
                    continue
                return got
            except ValueError:
                log_warn(
                    f"Got {got}, but it must be a number", parameters=self.parameters
                )
                continue

    def test_sizes(self, plot_func: Callable[[], None]) -> None:
        """
        Interactively test different font size parameters for a plot.

        Renders the plot, displays it, then prompts the user to either accept the
        current sizes or enter new values for each size parameter. Repeats until
        the user accepts.

        :param plot_func: A zero-argument callable that creates a matplotlib plot.
        :type plot_func: Callable[[], None]
        """
        done = False
        while not done:
            log_info(f"Plot with sizes: ", parameters=self.size_params)
            log_dict(self.size_params, n_indents=1, parameters=self.parameters)
            plot_func()
            plt.show()
            keepgoing = input(
                "Do you want to keep trying different sizes? (only y will keep going):"
            )
            if keepgoing.lower().strip() == "y":
                for key in self.size_params:
                    self.size_params[key] = self.get_size_input_number(key)
                self.set_size_parameters_from_dict(self.size_params)
            else:
                done = True
                break
        return

    def show(self, save_path: Optional[str] = None) -> None:
        """
        Show and/or save the current matplotlib figure.

        If ``save_path`` is provided, the figure is saved as both ``.pdf`` and
        ``.png``. If ``save_path`` is not already within ``figure_dir``, it is
        joined to ``figure_dir`` automatically. Whether the figure is also
        displayed interactively depends on the ``figure_skip_show`` parameter.
        If no ``save_path`` is given, the figure is shown immediately.

        :param save_path: Path (or filename) to save the figure to, without extension.
            If None, the figure is shown without saving.
        :type save_path: str or None
        """
        if save_path is not None:
            if os.path.abspath(save_path).startswith(
                os.path.abspath(self.parameters["figure_dir"])
            ):
                figure_path = save_path
            else:
                figure_path = self.parameters["figure_dir"] + f"/{save_path}"
        else:
            figure_path = None
        if figure_path is not None:
            figure_dir = os.path.dirname(figure_path)
            if not os.path.exists(figure_dir):
                os.makedirs(figure_dir)
            plt.savefig(f"{figure_path}.pdf")
            plt.savefig(f"{figure_path}.png")
            log_info(f"Saved figure to {figure_path}.pdf", parameters=self.parameters)
            if not self.parameters["figure_skip_show"]:
                plt.show()
            else:
                plt.clf()
        else:
            plt.show()
        return

    def get_stacked_bar_plot_func(
        self,
        df: pd.DataFrame,
        x_col: str,
        stacked_cols: list[str],
        colours: list[Any],
        skip_col: Optional[str] = None,
        skip_text_y_dip: float = 15,
        skip_text_rotation: float = 20,
        x_tick_rotation: float = 45,
        y_label: str = "Percentage (%)",
    ) -> Callable[[], None]:
        """
        Return a function that creates a stacked percentage bar plot.

        The returned callable takes no arguments and renders the plot using the
        dataframe and column configuration provided here. Rows must be sorted in
        ``[skip_col order, x_col order]`` for the grouping and spacing to render
        correctly.

        :param df: A pandas DataFrame containing the plot data.
        :type df: pd.DataFrame
        :param x_col: Column name to use for x-axis tick labels.
        :type x_col: str
        :param stacked_cols: Ordered list of column names to stack in each bar.
        :type stacked_cols: list[str]
        :param colours: List of colours corresponding to each entry in ``stacked_cols``.
        :type colours: list[Any]
        :param skip_col: Optional column name used to group bars with a gap between groups.
            Group labels are rendered below the x-axis.
        :type skip_col: str or None
        :param skip_text_y_dip: Vertical offset (downward) for group label text below the x-axis.
        :type skip_text_y_dip: float
        :param skip_text_rotation: Rotation angle in degrees for group label text.
        :type skip_text_rotation: float
        :param x_tick_rotation: Rotation angle in degrees for x-axis tick labels.
        :type x_tick_rotation: float
        :param y_label: Label for the y-axis.
        :type y_label: str
        :return: A zero-argument callable that renders the stacked bar plot.
        :rtype: Callable[[], None]
        """
        parameters = self.parameters
        # error out if any of the stacked_cols are not in the dataframe, if the length of colours is not the same as the length of stacked_cols, or if x_col / skip_col is not in the dataframe
        for col in (
            stacked_cols + [x_col] + ([skip_col] if skip_col is not None else [])
        ):
            if col not in df.columns:
                log_error(
                    f"Column {col} is not in the dataframe with columns {list(df.columns)}. Please check your input.",
                    parameters=parameters,
                )
        if len(colours) != len(stacked_cols):
            log_error(
                f"Length of colours {len(colours)} does not match length of stacked_cols {len(stacked_cols)}. Please check your input.",
                parameters=parameters,
            )

        def plot_func():
            fig, ax = plt.subplots(figsize=(12, 6))

            x_positions = []
            current_x = 0
            last_skip = None

            for i, row in df.iterrows():
                # Add a larger gap if the Method changes
                if skip_col is not None:
                    if last_skip is not None and row[skip_col] != last_skip:
                        current_x += 1.0  # The "Small Whitespace" between methods

                x_positions.append(current_x)

                # 3. Plotting the Stack
                bottom = 0
                for col, color in zip(stacked_cols, colours):
                    val = row[col]
                    ax.bar(
                        current_x,
                        val,
                        bottom=bottom,
                        color=color,
                        edgecolor="white",
                        width=0.8,
                    )
                    bottom += val
                if skip_col is not None:
                    last_skip = row[skip_col]
                current_x += 1  # Distance between x_cols within a skip_col

            # 4. Styling the X-Axis
            # ax.xaxis.tick_top()  # Moves ticks to the top
            # ax.xaxis.set_label_position("top")
            ax.set_xticks(x_positions)
            # Combine Model and Method for labels, or just Model
            ax.set_xticklabels(df[x_col], rotation=x_tick_rotation, ha="right")

            # Optional: Add Method labels below the models
            if skip_col is not None:
                df["x_pos"] = x_positions
                for method, group in df.groupby(skip_col, sort=False):
                    mid_point = group["x_pos"].mean()
                    ax.text(
                        mid_point,
                        -skip_text_y_dip,
                        method,
                        ha="center",
                        fontweight="bold",
                        rotation=skip_text_rotation,
                        fontsize=self.size_params["labels_font_size"],
                    )

            ax.set_ylabel(y_label)
            ax.legend(
                stacked_cols,
                loc="lower left",
                bbox_to_anchor=(0, 1, 1, 0.1),  # (x, y, width, height)
                mode="expand",
                ncol=len(stacked_cols),
                borderaxespad=0,
                frameon=False,
                fontsize=self.size_params["legend_font_size"],
            )
            plt.tight_layout(rect=[0, 0, 1, 0.95])

        return plot_func
