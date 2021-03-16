# RetroGrad
Minimalist pedagogical automatic differentiation in Scala.

Forward and reverse mode are supported. For reverse mode `Dual` nodes are assembled into a graph with gradients accessed via the `grad` method.

Example use:

```
// Reverse example:
val x = Dual(0.7)
val y = Dual(5.5)
val evaluation = x * y + x.sin
val xGradient = x.grad
val yGradient = y.grad
println(s"Evaluated expression ${evaluation.a}, gradients $xGradient, $yGradient")
/* Evaluated expression 4.494217687237691, gradients 6.264842187284488, 0.7
* */
```
