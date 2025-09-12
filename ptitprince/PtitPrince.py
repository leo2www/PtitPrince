from __future__ import division
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

__all__ = ["half_violinplot", "stripplot", "RainCloud"]
__version__ = '0.2.6'


# 替代 seaborn 内部类的基础实现
class _CategoricalPlotter:
    """Base class for categorical plots."""
    
    def __init__(self):
        self.plot_data = []
        self.plot_hues = None
        self.hue_names = None
        self.group_names = []
        self.colors = []
        self.gray = "0.3"
        self.orient = "v"
    
    def establish_variables(self, x, y, hue_data, data, orient, order, hue_order):
        """Extract and organize plotting variables from inputs."""
        # 简化的变量建立逻辑
        if data is not None:
            if isinstance(x, str):
                x_data = data[x]
            else:
                x_data = x
            if isinstance(y, str):
                y_data = data[y]
            else:
                y_data = y
            if isinstance(hue_data, str) and hue_data is not None:
                hue_data = data[hue_data]
            else:
                hue_data = hue_data
        else:
            x_data = x
            y_data = y
            hue_data = hue_data
        
        # 确定方向
        if orient is None:
            orient = "v"
        self.orient = orient
        
        # 处理分类和数值变量
        if orient == "v":
            self.categorical = x_data
            self.continuous = y_data
        else:
            self.categorical = y_data
            self.continuous = x_data
            
        # 设置组名
        if self.categorical is not None:
            if order is not None:
                self.group_names = list(order)
            else:
                unique_vals = pd.Series(self.categorical).dropna().unique()
                # 尝试数值排序，如果失败则用字符串排序
                try:
                    self.group_names = sorted(unique_vals, key=float)
                except (ValueError, TypeError):
                    self.group_names = sorted(unique_vals.astype(str))
        else:
            self.group_names = [""]
            
        # 设置色调名
        if hue_data is not None:
            if hue_order is not None:
                self.hue_names = list(hue_order)
            else:
                unique_hues = pd.Series(hue_data).dropna().unique()
                try:
                    self.hue_names = sorted(unique_hues, key=float)
                except (ValueError, TypeError):
                    self.hue_names = sorted(unique_hues.astype(str))
        else:
            self.hue_names = None
            
        # 组织绘图数据
        self.plot_data = []
        self.plot_hues = []
        
        if self.categorical is not None:
            for i, group in enumerate(self.group_names):
                group_mask = pd.Series(self.categorical) == group
                group_data = pd.Series(self.continuous)[group_mask].dropna().values
                
                if hue_data is not None:
                    group_hues = pd.Series(hue_data)[group_mask].values
                    # 过滤掉NaN值对应的hue
                    valid_mask = ~pd.isna(group_hues)
                    group_data = group_data[valid_mask]
                    group_hues = group_hues[valid_mask]
                    self.plot_hues.append(group_hues)
                else:
                    self.plot_hues.append(None)
                    
                self.plot_data.append(group_data)
        else:
            # 没有分类变量的情况
            group_data = pd.Series(self.continuous).dropna().values
            self.plot_data.append(group_data)
            self.plot_hues.append(None)

        if hue_data is None:
            self.plot_hues = None
    
    def establish_colors(self, color, palette, saturation):
        """Set up color mapping."""
        if self.hue_names is not None:
            n_colors = len(self.hue_names)
        else:
            n_colors = len(self.group_names)
            
        if palette is not None:
            colors = sns.color_palette(palette, n_colors)
        elif color is not None:
            colors = [color] * n_colors
        else:
            colors = sns.color_palette(n_colors=n_colors)
            
        # 应用饱和度
        if saturation < 1:
            colors = [sns.desaturate(color, saturation) for color in colors]
            
        self.colors = colors
    
    def annotate_axes(self, ax):
        """设置坐标轴标签"""
        pass


class _CategoricalScatterPlotter(_CategoricalPlotter):
    """Scatter plot with categorical organization."""
    
    def __init__(self):
        super().__init__()
        self.point_colors = []
        self.hue_offsets = []
    
    def establish_variables(self, x, y, hue, data, orient, order, hue_order):
        super().establish_variables(x, y, hue, data, orient, order, hue_order)
        
        # 设置偏移量
        if self.hue_names is not None:
            n_hues = len(self.hue_names)
            dodge_extent = 0.8 / n_hues
            self.hue_offsets = np.linspace(-dodge_extent/2, dodge_extent/2, n_hues)
        else:
            self.hue_offsets = [0]
            
        # 设置点颜色
        self.point_colors = []
        for i, group_data in enumerate(self.plot_data):
            if self.hue_names is not None and self.plot_hues is not None:
                group_colors = []
                for hue_val in self.plot_hues[i]:
                    try:
                        hue_idx = self.hue_names.index(hue_val)
                        group_colors.append(hue_idx)
                    except ValueError:
                        group_colors.append(0)
                self.point_colors.append(np.array(group_colors))
            else:
                self.point_colors.append(np.array([i] * len(group_data)))
    
    def add_legend_data(self, ax):
        """添加图例数据"""
        pass


class _StripPlotter(_CategoricalScatterPlotter):
    """1-d scatterplot with categorical organization."""
    def __init__(self, x, y, hue, data, order, hue_order,
                 jitter, dodge, orient, color, palette, width, move):
        """Initialize the plotter."""
        super().__init__()
        self.establish_variables(x, y, hue, data, orient, order, hue_order)
        self.establish_colors(color, palette, 1)

        # Set object attributes
        self.dodge = dodge
        self.width = width
        self.move = move

        if jitter == 1:  # Use a good default for `jitter = True`
            jlim = 0.1
        else:
            jlim = float(jitter)
        if self.hue_names is not None and dodge:
            jlim /= len(self.hue_names)
        self.jitterer = stats.uniform(-jlim, jlim * 2).rvs

    def draw_stripplot(self, ax, kws):
        """Draw the points onto `ax`."""
        palette = np.asarray(self.colors)
        for i, group_data in enumerate(self.plot_data):
            if self.plot_hues is None or not self.dodge:

                if self.hue_names is None:
                    hue_mask = np.ones(group_data.size, bool)
                else:
                    hue_mask = np.array([h in self.hue_names
                                         for h in self.plot_hues[i]], bool)

                strip_data = group_data[hue_mask]
                point_colors = np.asarray(self.point_colors[i])[hue_mask]

                # Plot the points in centered positions
                cat_pos = self.move + np.ones(strip_data.size) * i
                cat_pos += self.jitterer(len(strip_data))
                kws.update(c=palette[point_colors])

                if self.orient == "v":
                    ax.scatter(cat_pos, strip_data, **kws)
                else:
                    ax.scatter(strip_data, cat_pos, **kws)

            else:
                offsets = self.hue_offsets
                for j, hue_level in enumerate(self.hue_names):
                    hue_mask = np.array(self.plot_hues[i]) == hue_level
                    strip_data = group_data[hue_mask]

                    point_colors = np.asarray(self.point_colors[i])[hue_mask]
                    # Plot the points in centered positions
                    center = i + offsets[j]
                    cat_pos = self.move + np.ones(strip_data.size) * center
                    cat_pos += self.jitterer(len(strip_data))
                    kws.update(c=palette[point_colors])
                    if self.orient == "v":
                        ax.scatter(cat_pos, strip_data, **kws)
                    else:
                        ax.scatter(strip_data, cat_pos, **kws)

    def plot(self, ax, kws):
        """Make the plot."""
        self.draw_stripplot(ax, kws)
        self.add_legend_data(ax)
        self.annotate_axes(ax)


class _Half_ViolinPlotter(_CategoricalPlotter):

    def __init__(self, x, y, hue, data, order, hue_order,
                 bw, cut, scale, scale_hue, gridsize,
                 width, inner, split, dodge, orient, linewidth,
                 color, palette, saturation, offset):

        super().__init__()
        self.establish_variables(x, y, hue, data, orient, order, hue_order)
        self.establish_colors(color, palette, saturation)
        self.estimate_densities(bw, cut, scale, scale_hue, gridsize)

        self.gridsize = gridsize
        self.width = width
        self.dodge = dodge
        self.offset = offset

        if inner is not None:
            if not any([inner.startswith("quart"),
                        inner.startswith("box"),
                        inner.startswith("stick"),
                        inner.startswith("point")]):
                err = "Inner style '{}' not recognized".format(inner)
                raise ValueError(err)
        self.inner = inner

        if split and self.hue_names is not None and len(self.hue_names) < 2:
            msg = "There must be at least two hue levels to use `split`."
            raise ValueError(msg)
        self.split = split

        if linewidth is None:
            linewidth = mpl.rcParams["lines.linewidth"]
        self.linewidth = linewidth

    def estimate_densities(self, bw, cut, scale, scale_hue, gridsize):
        """Find the support and density for all of the data."""
        # Initialize data structures to keep track of plotting data
        if self.hue_names is None:
            support = []
            density = []
            counts = np.zeros(len(self.plot_data))
            max_density = np.zeros(len(self.plot_data))
        else:
            support = [[] for _ in self.plot_data]
            density = [[] for _ in self.plot_data]
            size = len(self.group_names), len(self.hue_names)
            counts = np.zeros(size)
            max_density = np.zeros(size)

        for i, group_data in enumerate(self.plot_data):

            # Option 1: we have a single level of grouping
            # --------------------------------------------

            if self.plot_hues is None:

                # Strip missing datapoints - 使用 pandas 替代 sns.utils.remove_na
                kde_data = pd.Series(group_data).dropna().values

                # Handle special case of no data at this level
                if kde_data.size == 0:
                    support.append(np.array([]))
                    density.append(np.array([1.]))
                    counts[i] = 0
                    max_density[i] = 0
                    continue

                # Handle special case of a single unique datapoint
                elif np.unique(kde_data).size == 1:
                    support.append(np.unique(kde_data))
                    density.append(np.array([1.]))
                    counts[i] = 1
                    max_density[i] = 0
                    continue

                # Fit the KDE and get the used bandwidth size
                kde, bw_used = self.fit_kde(kde_data, bw)

                # Determine the support grid and get the density over it
                support_i = self.kde_support(kde_data, bw_used, cut, gridsize)
                density_i = kde.evaluate(support_i)

                # Update the data structures with these results
                support.append(support_i)
                density.append(density_i)
                counts[i] = kde_data.size
                max_density[i] = density_i.max()

            # Option 2: we have nested grouping by a hue variable
            # ---------------------------------------------------

            else:
                for j, hue_level in enumerate(self.hue_names):

                    # Handle special case of no data at this category level
                    if not group_data.size:
                        support[i].append(np.array([]))
                        density[i].append(np.array([1.]))
                        counts[i, j] = 0
                        max_density[i, j] = 0
                        continue

                    # Select out the observations for this hue level
                    hue_mask = np.array(self.plot_hues[i]) == hue_level

                    # Strip missing datapoints - 使用 pandas 替代 sns.utils.remove_na
                    kde_data = pd.Series(group_data[hue_mask]).dropna().values

                    # Handle special case of no data at this level
                    if kde_data.size == 0:
                        support[i].append(np.array([]))
                        density[i].append(np.array([1.]))
                        counts[i, j] = 0
                        max_density[i, j] = 0
                        continue

                    # Handle special case of a single unique datapoint
                    elif np.unique(kde_data).size == 1:
                        support[i].append(np.unique(kde_data))
                        density[i].append(np.array([1.]))
                        counts[i, j] = 1
                        max_density[i, j] = 0
                        continue

                    # Fit the KDE and get the used bandwidth size
                    kde, bw_used = self.fit_kde(kde_data, bw)

                    # Determine the support grid and get the density over it
                    support_ij = self.kde_support(kde_data, bw_used,
                                                  cut, gridsize)
                    density_ij = kde.evaluate(support_ij)

                    # Update the data structures with these results
                    support[i].append(support_ij)
                    density[i].append(density_ij)
                    counts[i, j] = kde_data.size
                    max_density[i, j] = density_ij.max()

        # Scale the height of the density curve.
        if scale == "area":
            self.scale_area(density, max_density, scale_hue)
        elif scale == "width":
            self.scale_width(density)
        elif scale == "count":
            self.scale_count(density, counts, scale_hue)
        else:
            raise ValueError("scale method '{}' not recognized".format(scale))

        # Set object attributes that will be used while plotting
        self.support = support
        self.density = density

    def fit_kde(self, x, bw):
        """Estimate a KDE for a vector of data with flexible bandwidth."""
        # 现代版本的 scipy 支持灵活带宽
        kde = stats.gaussian_kde(x, bw_method=bw)
        
        # Extract the numeric bandwidth from the KDE object
        bw_used = kde.factor
        
        # Get the actual bandwidth
        bw_used = bw_used * x.std(ddof=1)
        
        return kde, bw_used

    def kde_support(self, x, bw, cut, gridsize):
        """Define a grid of support for the violin."""
        support_min = x.min() - bw * cut
        support_max = x.max() + bw * cut
        return np.linspace(support_min, support_max, gridsize)

    def scale_area(self, density, max_density, scale_hue):
        """Scale the relative area under the KDE curve."""
        if self.hue_names is None:
            for d in density:
                if d.size > 1:
                    d /= max_density.max()
        else:
            for i, group in enumerate(density):
                for d in group:
                    if scale_hue:
                        max_val = max_density[i].max()
                    else:
                        max_val = max_density.max()
                    if d.size > 1:
                        d /= max_val

    def scale_width(self, density):
        """Scale each density curve to the same height."""
        if self.hue_names is None:
            for d in density:
                if d.max() > 0:
                    d /= d.max()
        else:
            for group in density:
                for d in group:
                    if d.max() > 0:
                        d /= d.max()

    def scale_count(self, density, counts, scale_hue):
        """Scale each density curve by the number of observations."""
        if self.hue_names is None:
            if counts.max() == 0:
                return
            for count, d in zip(counts, density):
                if d.max() > 0:
                    d /= d.max()
                    d *= count / counts.max()
        else:
            for i, group in enumerate(density):
                for j, d in enumerate(group):
                    if counts[i].max() == 0:
                        continue
                    count = counts[i, j]
                    if scale_hue:
                        scaler = count / counts[i].max()
                    else:
                        scaler = count / counts.max()
                    if d.max() > 0:
                        d /= d.max()
                        d *= scaler

    @property
    def dwidth(self):
        if self.hue_names is None or not self.dodge:
            return self.width / 2
        elif self.split:
            return self.width / 2
        else:
            return self.width / (2 * len(self.hue_names))

    def draw_violins(self, ax, kws):
        """Draw the violins onto `ax`."""
        fill_func = ax.fill_betweenx if self.orient == "v" else ax.fill_between
        for i, group_data in enumerate(self.plot_data):

            kws.update(dict(edgecolor=self.gray, linewidth=self.linewidth))

            if self.plot_hues is None:
                support, density = self.support[i], self.density[i]

                if support.size == 0:
                    continue
                elif support.size == 1:
                    val = np.ndarray.item(support)
                    d = np.ndarray.item(density)
                    self.draw_single_observation(ax, i, val, d)
                    continue

                grid = np.ones(self.gridsize) * i
                fill_func(support,
                          -self.offset + grid - density * self.dwidth,
                          -self.offset + grid,
                          facecolor=self.colors[i],
                          **kws)

                if self.inner is None:
                    continue

                violin_data = pd.Series(group_data).dropna().values

                if self.inner.startswith("box"):
                    self.draw_box_lines(ax, violin_data, support, density, i)
                elif self.inner.startswith("quart"):
                    self.draw_quartiles(ax, violin_data, support, density, i)
                elif self.inner.startswith("stick"):
                    self.draw_stick_lines(ax, violin_data, support, density, i)
                elif self.inner.startswith("point"):
                    self.draw_points(ax, violin_data, i)

            else:
                offsets = np.linspace(-0.4, 0.4, len(self.hue_names))
                for j, hue_level in enumerate(self.hue_names):
                    support, density = self.support[i][j], self.density[i][j]
                    kws["facecolor"] = self.colors[j]

                    if support.size == 0:
                        continue
                    elif support.size == 1:
                        val = np.ndarray.item(support)
                        d = np.ndarray.item(density)
                        if self.split:
                            d = d / 2
                        at_group = i + offsets[j]
                        self.draw_single_observation(ax, at_group, val, d)
                        continue

                    if self.split:
                        grid = np.ones(self.gridsize) * i
                        if j:
                            fill_func(support,
                                      -self.offset + grid,
                                      -self.offset + grid + density * self.dwidth,
                                      **kws)
                        else:
                            fill_func(support,
                                      -self.offset + grid - density * self.dwidth,
                                      -self.offset + grid,
                                      **kws)
                    else:
                        grid = np.ones(self.gridsize) * (i + offsets[j])
                        fill_func(support,
                                  -self.offset + grid - density * self.dwidth,
                                  -self.offset + grid,
                                  **kws)

    def draw_single_observation(self, ax, at_group, at_quant, density):
        """Draw a line to mark a single observation."""
        d_width = density * self.dwidth
        if self.orient == "v":
            ax.plot([at_group - d_width, at_group + d_width],
                    [at_quant, at_quant],
                    color=self.gray,
                    linewidth=self.linewidth)
        else:
            ax.plot([at_quant, at_quant],
                    [at_group - d_width, at_group + d_width],
                    color=self.gray,
                    linewidth=self.linewidth)

    def draw_box_lines(self, ax, data, support, density, center):
        """Draw boxplot information at center of the density."""
        q25, q50, q75 = np.percentile(data, [25, 50, 75])
        whisker_lim = 1.5 * stats.iqr(data)
        h1 = np.min(data[data >= (q25 - whisker_lim)])
        h2 = np.max(data[data <= (q75 + whisker_lim)])

        if self.orient == "v":
            ax.plot([center, center], [h1, h2],
                    linewidth=self.linewidth,
                    color=self.gray)
            ax.plot([center, center], [q25, q75],
                    linewidth=self.linewidth * 3,
                    color=self.gray)
            ax.scatter(center, q50,
                       zorder=3,
                       color="white",
                       edgecolor=self.gray,
                       s=np.square(self.linewidth * 2))
        else:
            ax.plot([h1, h2], [center, center],
                    linewidth=self.linewidth,
                    color=self.gray)
            ax.plot([q25, q75], [center, center],
                    linewidth=self.linewidth * 3,
                    color=self.gray)
            ax.scatter(q50, center,
                       zorder=3,
                       color="white",
                       edgecolor=self.gray,
                       s=np.square(self.linewidth * 2))

    def draw_quartiles(self, ax, data, support, density, center, split=False):
        """Draw the quartiles as lines at width of density."""
        q25, q50, q75 = np.percentile(data, [25, 50, 75])

        self.draw_to_density(ax, center, q25, support, density, split,
                             linewidth=self.linewidth,
                             dashes=[self.linewidth * 1.5] * 2)
        self.draw_to_density(ax, center, q50, support, density, split,
                             linewidth=self.linewidth,
                             dashes=[self.linewidth * 3] * 2)
        self.draw_to_density(ax, center, q75, support, density, split,
                             linewidth=self.linewidth,
                             dashes=[self.linewidth * 1.5] * 2)

    def draw_points(self, ax, data, center):
        """Draw individual observations as points at middle of the violin."""
        kws = dict(s=np.square(self.linewidth * 2),
                   color=self.gray,
                   edgecolor=self.gray)

        grid = np.ones(len(data)) * center

        if self.orient == "v":
            ax.scatter(grid, data, **kws)
        else:
            ax.scatter(data, grid, **kws)

    def draw_stick_lines(self, ax, data, support, density,
                         center, split=False):
        """Draw individual observations as sticks at width of density."""
        for val in data:
            self.draw_to_density(ax, center, val, support, density, split,
                                 linewidth=self.linewidth * .5)

    def draw_to_density(self, ax, center, val, support, density, split, **kws):
        """Draw a line orthogonal to the value axis at width of density."""
        idx = np.argmin(np.abs(support - val))
        width = self.dwidth * density[idx] * .99

        kws["color"] = self.gray

        if self.orient == "v":
            if split == "left":
                ax.plot([center - width, center], [val, val], **kws)
            elif split == "right":
                ax.plot([center, center + width], [val, val], **kws)
            else:
                ax.plot([center - width, center + width], [val, val], **kws)
        else:
            if split == "left":
                ax.plot([val, val], [center - width, center], **kws)
            elif split == "right":
                ax.plot([val, val], [center, center + width], **kws)
            else:
                ax.plot([val, val], [center - width, center + width], **kws)

    def plot(self, ax, kws):
        """Make the violin plot."""
        self.draw_violins(ax, kws)
        self.annotate_axes(ax)


# 其余函数保持不变...
def stripplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None,
              jitter=True, dodge=False, orient=None, color=None, palette=None, move=0,
              size=5, edgecolor="gray", linewidth=0, ax=None, width=.8, **kwargs):

    if "split" in kwargs:
        dodge = kwargs.pop("split")
        msg = "The `split` parameter has been renamed to `dodge`."
        warnings.warn(msg, UserWarning)

    plotter = _StripPlotter(x, y, hue, data, order, hue_order,
                            jitter, dodge, orient, color, palette, width, move)
    if ax is None:
        ax = plt.gca()

    kwargs.setdefault("zorder", 3)
    size = kwargs.get("s", size)
    if linewidth is None:
        linewidth = size / 10
    if edgecolor == "gray":
        edgecolor = plotter.gray
    kwargs.update(dict(s=size ** 2,
                       edgecolor=edgecolor,
                       linewidth=linewidth))

    plotter.plot(ax, kwargs)
    return ax


def half_violinplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None,
                    bw="scott", cut=2, scale="area", scale_hue=True, gridsize=100,
                    width=.8, inner="box", split=False, dodge=True, orient=None,
                    linewidth=None, color=None, palette=None, saturation=.75,
                    ax=None, offset=.15, **kwargs):

    plotter = _Half_ViolinPlotter(x, y, hue, data, order, hue_order,
                                  bw, cut, scale, scale_hue, gridsize,
                                  width, inner, split, dodge, orient, linewidth,
                                  color, palette, saturation, offset)

    if ax is None:
        ax = plt.gca()

    plotter.plot(ax, kwargs)
    return ax


def RainCloud(x=None, y=None, hue=None, data=None,
              order=None, hue_order=None,
              orient="v", width_viol=.7, width_box=.15,
              palette="Set2", bw=.2, linewidth=1, cut=0.,
              scale="area", jitter=1, move=0., offset=None,
              point_size=3, ax=None, pointplot=False,
              alpha=None, dodge=False, linecolor='red', **kwargs):

    if orient == 'h':  # swap x and y
        x, y = y, x
    if ax is None:
        ax = plt.gca()

    if offset is None:
        offset = max(width_box/1.8, .15) + .05
    n_plots = 3
    split = False
    boxcolor = "black"
    boxprops = {'facecolor': 'none', "zorder": 10}
    if hue is not None:
        split = True
        boxcolor = palette
        boxprops = {"zorder": 10}

    kwcloud = dict()
    kwbox = dict(saturation=1, whiskerprops={'linewidth': 2, "zorder": 10})
    kwrain = dict(zorder=0, edgecolor="white")
    kwpoint = dict(capsize=0., errwidth=0., zorder=20)
    
    for key, value in kwargs.items():
        if "cloud_" in key:
            kwcloud[key.replace("cloud_", "")] = value
        elif "box_" in key:
            kwbox[key.replace("box_", "")] = value
        elif "rain_" in key:
            kwrain[key.replace("rain_", "")] = value
        elif "point_" in key:
            kwpoint[key.replace("point_", "")] = value
        else:
            kwcloud[key] = value

    # Draw cloud/half-violin
    half_violinplot(x=x, y=y, hue=hue, data=data,
                    order=order, hue_order=hue_order,
                    orient=orient, width=width_viol,
                    inner=None, palette=palette, bw=bw, linewidth=linewidth,
                    cut=cut, scale=scale, split=split, offset=offset, ax=ax, **kwcloud)

    # Draw umberella/boxplot
    if hue is None:
        sns.boxplot(x=x, y=y, data=data, orient=orient, width=width_box,
                    order=order, color=boxcolor, showcaps=True, boxprops=boxprops,
                    dodge=dodge, ax=ax, **kwbox)
    else:
        sns.boxplot(x=x, y=y, hue=hue, data=data, orient=orient, width=width_box,
                    order=order, hue_order=hue_order,
                    showcaps=True, boxprops=boxprops,
                    palette=palette, dodge=dodge, ax=ax, **kwbox)

    # Set alpha of the two
    if alpha is not None:
        _ = plt.setp(ax.collections + ax.artists, alpha=alpha)

    # Draw rain/stripplot
    ax = stripplot(x=x, y=y, hue=hue, data=data, orient=orient,
                   order=order, hue_order=hue_order, palette=palette,
                   move=move, size=point_size, jitter=jitter, dodge=dodge,
                   width=width_box, ax=ax, **kwrain)

    # Add pointplot
    if pointplot:
        n_plots = 4
        if hue is not None:
            n_cat = len(np.unique(data[hue]))
            sns.pointplot(x=x, y=y, hue=hue, data=data,
                          orient=orient, order=order, hue_order=hue_order,
                          dodge=width_box * (1 - 1 / n_cat), palette=palette, ax=ax, **kwpoint)
        else:
            sns.pointplot(x=x, y=y, hue=hue, data=data, color=linecolor,
                          orient=orient, order=order, hue_order=hue_order,
                          dodge=width_box/2., ax=ax, **kwpoint)

    # Prune the legend, add legend title
    if hue is not None:
        handles, labels = ax.get_legend_handles_labels()
        _ = plt.legend(handles[0:len(labels)//n_plots], labels[0:len(labels)//n_plots],
                       bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                       title=str(hue))

    # 修复坐标轴调整逻辑 - 这里是问题所在
    # 注释掉或者修复这段代码，因为它会导致图形超出范围
    # if orient == "h":
    #     ylim = list(ax.get_ylim())
    #     ylim[-1] -= (width_box + width_viol)/4.
    #     _ = ax.set_ylim(ylim)
    # elif orient == "v":
    #     xlim = list(ax.get_xlim())
    #     xlim[-1] -= (width_box + width_viol)/4.
    #     _ = ax.set_xlim(xlim)

    # 改进的坐标轴调整方式 - 增加左边距和右边距
    if orient == "v":
        # 对于垂直方向，增加 x 轴的左右边距
        xlim = ax.get_xlim()
        x_range = xlim[1] - xlim[0]
        left_margin = x_range * 0.15   # 左边距增加到15%
        right_margin = x_range * 0.1   # 右边距10%
        ax.set_xlim(xlim[0] - left_margin, xlim[1] + right_margin)
    else:
        # 对于水平方向，增加 y 轴的上下边距
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        bottom_margin = y_range * 0.15  # 下边距增加到15%
        top_margin = y_range * 0.1      # 上边距10%
        ax.set_ylim(ylim[0] - bottom_margin, ylim[1] + top_margin)


    return ax
