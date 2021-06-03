package retrograd

import Op._

abstract class Dual(val a: Double, val b: Double = 1.0, val op: Op = IDT) {

	private val childrenBldr = Seq.newBuilder[Dual]

	private val parentsBldr = Seq.newBuilder[Dual]

	def children: Seq[Dual] = childrenBldr.result()

	def parents: Seq[Dual] = parentsBldr.result()

	def adjoint(v: Dual): Double

	private var checkpoint: Double = Double.NaN

	def grad: Double =
		if (checkpoint.isNaN) {
			val vjp =
				if (children.nonEmpty)
					children.map(_.adjoint(this)).sum
				else 1.0
			checkpoint = vjp
			vjp
		} else checkpoint

	def + (j: Dual): Dual = {
		val out = Dual(a + j.a, op = ADD)
		updateAdjacent2(out, j)
		out
	}

	def * (j: Dual): Dual = {
		val out = Dual(a * j.a, op = MUL)
		updateAdjacent2(out, j)
		out
	}

	def / (j: Dual): Dual = {
		val out = Dual(a / j.a, op = DIV)
		updateAdjacent2(out, j)
		out
	}

	def sin: Dual = {
		val out = Dual(math.sin(a), op = SIN)
		updateAdjacent1(out)
		out
	}

	def exp: Dual = {
		val out = Dual(math.exp(a), op = EXP)
		updateAdjacent1(out)
		out
	}

	private def updateAdjacent1(out: Dual): Unit = {
		out.parentsBldr += this
		childrenBldr    += out
	}

	private def updateAdjacent2(out: Dual, j: Dual): Unit = {
		out.parentsBldr += this
		out.parentsBldr += j
		childrenBldr    += out
		j.childrenBldr  += out
	}

	def asTuple: (Double, Double) = (a, b)

	def copy(a: Double = this.a, b: Double = this.b, op: Op = this.op): Dual = Dual(a, b, op)
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