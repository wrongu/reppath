# Representational Paths

## Usage

#### Compute the total slack of a representational path

```python
from reppath.slack import slack

path = ... # see repsim package

print(slack(path, method='angle'))
```

#### Draw the clump-plot of a representational path

```python
from reppath.draw import draw_clumping_diagram

model_a = ... # see repsim package
model_b = ... # see repsim package

draw_clumping_diagram([model_a, model_b])
```

![](docs/img/clump-sketch.png)
