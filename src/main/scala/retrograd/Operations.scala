package retrograd

object Op extends Enumeration {
	type Op = Value
	val IDT, DIV, MUL, ADD, SUB, SIN, EXP = Value
}

object Primitives {
	import Op._

	def arity = Map(
		IDT -> 0,
		MUL -> 2,
		SIN -> 1,
		ADD -> 2,
		DIV -> 2,
		EXP -> 1,
		SUB -> 2
	)

	def funcMap2 = Map(
		MUL -> ((a: Dual, b: Dual) => multiply(a, b)),
		DIV -> ((a: Dual, b: Dual) => divide(a, b)),
		SUB -> ((a: Dual, b: Dual) => subtract(a, b)),
		ADD -> ((a: Dual, b: Dual) => add(a, b))
	)

	def funcMap1 = Map(
		SIN -> ((a: Dual) => sin(a)),
		EXP -> ((a: Dual) => exp(a))
	)

	def add(atuple: Dual, btuple: Dual): Dual = {
		val (a, adot) = atuple.asTuple
		val (b, bdot) = btuple.asTuple
		Dual(a + b, adot + bdot)
	}

	def subtract(atuple: Dual, btuple: Dual): Dual = {
		val (a, adot) = atuple.asTuple
		val (b, bdot) = btuple.asTuple
		Dual(a - b, adot - bdot)
	}

	def multiply(atuple: Dual, btuple: Dual): Dual = {
		val (a, adot) = atuple.asTuple
		val (b, bdot) = btuple.asTuple
		Dual(a * b, adot * b + bdot * a)
	}

	def divide(atuple: Dual, btuple: Dual): Dual = {
		val (a, adot) = atuple.asTuple
		val (b, bdot) = btuple.asTuple
		Dual(a / b, (adot * b - bdot * a) / (b * b))
	}

	def exp(atuple: Dual): Dual = {
		val (a, adot) = atuple.asTuple
		Dual(math.exp(a), math.exp(a) * adot)
	}

	def sin(atuple : Dual): Dual = {
		val (a, adot) = atuple.asTuple
		Dual(math.sin(a), math.cos(a) * adot)
	}
}
