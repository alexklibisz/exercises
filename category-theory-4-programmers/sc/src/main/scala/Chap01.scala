object Chap01 extends App {

  // Implement the identity function. Technically alredy available as `identity`.
  def id[A](a: A): A = a

  // Implement the composition function.
  // Note the `compose` is already defined on Function1.
  // My version of compo is basically the same as `andThen` on Function1.
  implicit class Composable[A, B](f: A => B) {
    def comp[C](g: B => C): A => C = x => g(f(x))
  }

  // Test composition.
  val addPi: Int => Double = _ + Math.PI
  val stringify: Double => String = _.toString
  val composed: Int => String = addPi comp stringify

  println(addPi(2))
  println(stringify(Math.PI))
  println(composed(2))
  println((addPi comp stringify)(2))

  // Test the composition respects identity.
  val f = addPi
  val idCompF = id[Int] _ comp f
  val fCompId = f comp id

  // You can only really test for concrete values and compare the classes. Both of these are equivalent.
  // But that doesn't really tell you that composition in general will respect identity.
  println(idCompF(2) == fCompId(2))
  println(idCompF.getClass == fCompId.getClass)

}
