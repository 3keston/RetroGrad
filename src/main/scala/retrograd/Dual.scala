package retrograd

import Op._

abstract class Dual(val a: Double, val b: Double = 1.0, val op: Op = IDT) {

	def grad: Double =
		if (checkpoint.isNaN) {
			val vjp =
				if (children.nonEmpty)
					children.map(_.adjoint(this)).sum
				else 1.0
			checkpoint = vjp
			vjp
		} else checkpoint

	def adjoint(v: Dual): Double

	def + (j: Dual): Dual = updateAdjacent2(Dual(a + j.a, op = ADD), j)

	def * (j: Dual): Dual = updateAdjacent2(Dual(a * j.a, op = MUL), j)

	def / (j: Dual): Dual = updateAdjacent2(Dual(a / j.a, op = DIV), j)

	def sin: Dual = updateAdjacent1(Dual(math.sin(a), op = SIN))

	def exp: Dual = updateAdjacent1(Dual(math.exp(a), op = EXP))

	def children: Seq[Dual] = childrenBldr.result()

	def parents: Seq[Dual] = parentsBldr.result()

	private var checkpoint: Double = Double.NaN

	private val childrenBldr = Seq.newBuilder[Dual]

	private val parentsBldr = Seq.newBuilder[Dual]

	private def updateAdjacent1(out: Dual): Dual = {
		out.parentsBldr += this
		childrenBldr    += out
		out
	}

	private def updateAdjacent2(out: Dual, j: Dual): Dual = {
		out.parentsBldr += this
		out.parentsBldr += j
		childrenBldr    += out
		j.childrenBldr  += out
		out
	}

	def asTuple: (Double, Double) = (a, b)

	def copy(a: Double = this.a,
	         b: Double = this.b,
	         op: Op    = this.op): Dual = Dual(a, b, op)
}

object Dual {
	import Primitives._
	def apply(a: Double, b: Double = 1.0, op: Op = IDT): Dual = {
		val opArity = arity(op)
		if (opArity == 0) new Dual(a, b, op) {
			def adjoint(v: Dual): Double = 1.0
		}
		else if (opArity == 1) new Dual(a, b, op) {
			def adjoint(v: Dual): Double = Primitives.funcMap1(op)(v.copy(b = grad)).b
		}
		else if (opArity == 2) new Dual(a, b, op) {
			def adjoint(v: Dual): Double = {
				val partials = parents.iterator.map { parent =>
					if (parent == v)
						v.copy(b = grad)
					else parent.copy(b = 0.0)
				}
				Primitives.funcMap2(op)(partials.next(), partials.next()).b
			}
		}
		else throw new Exception(s"Unsupported ${opArity}-ary $op")
	}
}