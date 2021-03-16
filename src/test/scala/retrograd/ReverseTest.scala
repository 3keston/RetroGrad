package retrograd

import Primitives._

import org.scalactic.Tolerance._
import org.scalatest.funsuite.AnyFunSuite

import scala.language.postfixOps

class ReverseTest extends AnyFunSuite {
	val epsilon = 1E-3

	test ("forward dual") {

		def func(x1: Dual, x2: Dual) = {
			val a = divide (x1 , x2)
			val b = exp(x2)
			multiply(
				subtract(
					add(sin(a), a),
					b),
				subtract(a, b))
		}

		val res = func(Dual(1.5 , 1.0) , Dual(0.5 , 0.0))
		println(res)
		assert(res.asTuple._1 === 2.0167 +- epsilon)
		assert(res.asTuple._2 === 3.0118 +- epsilon)
	}

	test("basic reverse auto diff test") {
		val x = Dual(a=0.5)
		val y = Dual(a=4.2)
		val xr = 0.5
		val yr = 4.2
		val ansr = xr * yr + math.sin(xr)

		val ans = x * y + x.sin
		println(s"cop ${ans.a} real $ansr x children ${ans.parents}")
		ans.parents.foreach(par => println(s"parent $par for node $ans"))
		println(s"inputs x ${x.children} and y ${y.children}")
		println(s"xhat ${x.grad}")
		println(s"yhat ${y.grad}")
		assert(x.grad === 5.0775 +- epsilon)
		assert(y.grad === 0.5 +- epsilon)
		assert(ans.a === ansr +- epsilon)
	}

	test("sigmoid test") {
		object SigmoidActivation {
			def activeFunc(d: Double): Double = 1.0 / (1 + scala.math.exp(-d))
			def derivActive(d: Double): Double = d * (1 - d)
		}

		val d = 3.5
		val fwSig = SigmoidActivation.activeFunc(d)
		val rvSig = SigmoidActivation.derivActive(fwSig)

		val one = Dual(1.0)
		val num = Dual(d)
		val neg = Dual(-1.0)

		val exp = one / (one + (num * neg).exp) // TODO: better method to negate a value, i.e. -d instead of d * -1
		assert(num.grad === rvSig +- epsilon)
	}
}