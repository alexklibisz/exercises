import scala.collection.mutable

object Chap02 extends App {

  def memoize[A, B](f: A => B): A => B = {
    val memo: mutable.Map[A, B] = scala.collection.mutable.Map.empty
    a: A => memo.get(a) match {
      case Some(b) => b
      case None =>
        val b = f(a)
        memo.synchronized(memo.update(a, b))
        b
    }
  }

  def fact(n: Int): Int = (1 until n).product
  val mfact = memoize(fact)

  val t0 = System.currentTimeMillis()
  println(mfact(10))
  println(System.currentTimeMillis() - t0)

  val t1 = System.currentTimeMillis()
  println(mfact(10))
  println(System.currentTimeMillis() - t1)

}
