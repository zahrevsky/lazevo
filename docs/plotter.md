<!-- markdownlint-disable -->

<a href="../lazevo/plotter.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `plotter`
This module contains functions for plotting and saving 2D slices of Universes using matplotlib. 

Those methods could be a part of Universe and UniverseTrajectory classes, but factoring them out helps to distribute responsibility, as plotting and minimizing action are two distinct tasks. 


---

<a href="../lazevo/plotter.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_projection_to_file`

```python
plot_projection_to_file(filename, particles, displacements=None)
```

Plots the 2D projection of the given particles and their displacements, and saves the plot to a file. 



**Args:**
 
 - <b>`filename`</b> (str):  the name of the file to save the plot to. 
 - <b>`particles`</b> (list of tuples):  a list of 2D tuples representing the particles to be plotted. 
 - <b>`displacements`</b> (list of tuples, optional):  a list of 2D tuples representing the displacements of the particles. If not provided, only the particles will be plotted. 


---

<a href="../lazevo/plotter.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `project`

```python
project(item_3d, axis=0)
```

Projects a 3D point onto a 2D plane, perpendicular to the given axis. 



**Args:**
 
 - <b>`item_3d`</b> (tuple):  a 3D tuple to be projected. 
 - <b>`axis`</b> (int, optional):  the axis to drop. The default is 0. 



**Returns:**
 
 - <b>`tuple`</b>:  a 2D tuple representing the projection of the 3D item. 


---

<a href="../lazevo/plotter.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_universe_trajectory_to_file`

```python
plot_universe_trajectory_to_file(
    filename,
    universe_trajectory,
    axis=0,
    start=0.1,
    end=0.1625
)
```

Plots the 2D projection of a slice of a universe trajectory, and saves the plot to a file. 

Similar to `plot_universe_to_file` but also draws displacements. 



**Args:**
 
 - <b>`filename`</b> (str):  the name of the file to save the plot to. 
 - <b>`universe_trajectory`</b> (lazevo.piza.UniverseTrajectory):  the universe trajectory to plot. 
 - <b>`axis`</b> (int, optional):  the axis to drop. The default is 0. 
 - <b>`start`</b> (float, optional):  the start of the slice. The default is 0.1. 
 - <b>`end`</b> (float, optional):  the end of the slice. The default is 0.1625. 


---

<a href="../lazevo/plotter.py#L83"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_universe_to_file`

```python
plot_universe_to_file(filename, universe, axis=0, start=0.1, end=0.1625)
```

Plots the 2D projection of a slice of a universe, and saves the plot to a file. 



**Args:**
 
 - <b>`filename`</b> (str):  the name of the file to save the plot to. 
 - <b>`universe`</b> (lazevo.piza.Universe):  the universe to plot. 
 - <b>`axis`</b> (int, optional):  the axis to drop. The default is 0. 
 - <b>`start`</b> (float, optional):  the start of the slice. The default is 0.1. 
 - <b>`end`</b> (float, optional):  the end of the slice. The default is 0.1625. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
