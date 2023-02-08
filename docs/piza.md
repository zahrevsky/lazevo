<!-- markdownlint-disable -->

<a href="../lazevo/piza.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `piza`
Provides PIZA algorithm implementation, as well as classes to represent data structures. 


---

<a href="../lazevo/piza.py#L164"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `piza`

```python
piza(realizations: list[piza.UniverseTrajectory], n_iters=10)
```

Minimizing action by Path Interchange Zeldovich Approximation (PIZA). 

Runs PIZA for several UniverseTrajectory objects 



**Args:**
 
 - <b>`realizations`</b>:  Each realization is a UniverseTrajectory with a unique initial positions 
 - <b>`n_iters`</b>:  Number of iterations to perform in the minimization process 


---

<a href="../lazevo/piza.py#L179"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `read_universe`

```python
read_universe(path)
```

Read the universe from the given file path. 



**Args:**
 
 - <b>`path`</b>:  Path to the file. 



**Returns:**
 A Universe instance, representing the universe read from the file. 


---

<a href="../lazevo/piza.py#L193"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `random_init_universe`

```python
random_init_universe(universe)
```

Generates random initial positions for each particle from the given Universe object. 



**Returns:**
 
 - <b>`List[Tuple[float, float, float]]`</b>:  List, containing the same number  of elements, as universe.particles does. Each element is a 3-item  tuple with random coordinates (x, y, z). Generated coordinates are  from range (universe.min_coords, universe.max_coords). So, no random  coordinate is larger or smaller than any given coordinate from Universe. 


---

<a href="../lazevo/piza.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Universe`
Universe class represents the universe of particles. 

Primarily purpose is to store data: particle positions and boundaries. It also has a method slice, to extract a sub-universe from the current universe. 

<a href="../lazevo/piza.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Universe.__init__`

```python
__init__(particles)
```








---

<a href="../lazevo/piza.py#L37"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Universe.slice`

```python
slice(axis: int, start: float, end: float)
```

Slices a universe along the given axis. 

Extract a sub-universe, containing all particles with value  from a range (start, end) for a given axis. 



**Args:**
 
 - <b>`axis`</b>:  Axis to slice on, either 0 (x-axis), 1 (y-axis), or 2 (z-axis). 
 - <b>`start`</b>:  Starting coordinate of the slice, normalized to the length universe takes along this coordinate. So, 0.0 corresponds to the minimal coordinate value, and 1.0 corresponds to maximal coordinate value. 
 - <b>`end`</b>:  End coordinate of the slice. Same as with start: the value is in the rage from 0 to 1. 



**Returns:**
 A new Universe instance, representing the sub-universe. 


---

<a href="../lazevo/piza.py#L57"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `UniverseTrajectory`
Does PIZA and represents the trajectory of all the particles in the universe. 

It holds information about the initial and actual positions of the particles  as well as an action of the system 

Unlike Universe, it not only holds information, but also allows to perform PIZA algorithm step. 



**Attributes:**
 
 - <b>`pairs`</b> (Dict[Tuple[float, float, float], Tuple[float, float, float]]):  A dictionary where keys are initial positions of particles and values are their final positions. 
 - <b>`min_coords`</b> (List[float, float, float]):  Minimum values of x, y and z coordinates. 
 - <b>`max_coords`</b> (List[float, float, float]):  Maximum values of x, y and z coordinates. 
 - <b>`sizes`</b> (List[float, float, float]):  Sizes of the universe along each axis. 
 - <b>`action`</b> (int):  The sum of each point's displacements squared, a. k. a. ∑ψᵢ². 



**Args:**
 
 - <b>`particles`</b>:  List of particles in the trajectory. 
 - <b>`init_positions`</b>:  List of initial positions of the particles.  

<a href="../lazevo/piza.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `UniverseTrajectory.__init__`

```python
__init__(particles, init_positions)
```






---

#### <kbd>property</kbd> UniverseTrajectory.displacements



**Returns:**
 
 - <b>`List[Tuple[float, float, float]]`</b>:  Displacement of each particle from its initial position to final position. 

---

#### <kbd>property</kbd> UniverseTrajectory.init_positions



**Returns:**
 
 - <b>`List[Tuple[float, float, float]]`</b>:  The initial positions of particles. 

---

#### <kbd>property</kbd> UniverseTrajectory.particles



**Returns:**
 
 - <b>`List[Tuple[float, float, float]]`</b>:  The final positions of particles. 



---

<a href="../lazevo/piza.py#L148"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `UniverseTrajectory.do_piza_step`

```python
do_piza_step()
```

Picks randomly two particles and swaps their initial positions, if it decreases total action. 

---

<a href="../lazevo/piza.py#L139"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `UniverseTrajectory.slice`

```python
slice(axis, start, end)
```

See Universe.slice. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
