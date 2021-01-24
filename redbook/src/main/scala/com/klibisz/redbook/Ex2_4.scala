package com.klibisz.redbook

object Ex2_4 {

  import Ex2_3.curry

  def uncurry[A, B, C](f: A => B => C): (A, B) => C = (a: A, b: B) => f(a)(b)

  def main(args: Array[String]): Unit = {
    val q: Int => Int => Int = curry[Int, Int, Int](_ + _)
    val r: (Int, Int) => Int = uncurry(q)
    println(q(1)(2))
    println(r(1, 2))
  }

}
