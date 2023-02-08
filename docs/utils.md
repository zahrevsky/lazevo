<!-- markdownlint-disable -->

<a href="../lazevo/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `utils`
Includes miscelaneous helpers: parse input parameters, calculated squared distance, etc. 


---

<a href="../lazevo/utils.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `parse_params`

```python
parse_params() → dict[str, Any]
```

Reads parameters from command line and configuration file. 



**Returns:**
  A dict with config parameters, that might be nested. For example: 

 {'paths':   {'input':  
 - <b>`{'particles_positions'`</b>:  'input/coord_LCDM.txt', 
 - <b>`'random_positions'`</b>:  None}, 'output':  
 - <b>`{'universe_fig'`</b>:  'output/slice.png', 
 - <b>`'trajectory_txt_prefix'`</b>:  'output/coord_LCDM_', 
 - <b>`'trajectory_fig_prefix'`</b>:  'output/displacements_'}}, 
 - <b>`'dillution_factor'`</b>:  1, 
 - <b>`'mode'`</b>:  'periodic', 
 - <b>`'n_realizations'`</b>:  8, 
 - <b>`'n_iters'`</b>:  10} 


---

<a href="../lazevo/utils.py#L58"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `dist_squared`

```python
dist_squared(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float]
) → float
```

Calculates squre of the distance between two points. 

Order of points is not important: 

 dist_squared(p1, p2) == dist_squared(p2, p1) 



**Args:**
 
 - <b>`p1`</b>:  Tuple of floats (x₁, y₁, z₁) 
 - <b>`p2`</b>:  Tuple of floats (x₂, y₂, z₂) 



**Returns:**
 (x₂—x₁)² + (y₂—y₁)² + (z₂—z₁)² 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
