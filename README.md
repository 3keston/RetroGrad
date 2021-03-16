# RetroGrad
Minimalist pedagogical automatic differentiation in Scala.

In reverse mode, `Dual` nodes are assembled into a graph with gradients accessed via the `grad` method. Forward mode is supported by directly composing primitive operations, e.g. `multiply(Dual(1.5 , 1.0) , Dual(0.5 , 0.0))` will compute the gradient wrt the first param.

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
