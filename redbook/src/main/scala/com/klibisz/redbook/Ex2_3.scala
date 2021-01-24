package com.klibisz.redbook

object Ex2_3 {

  def curry[A, B, C](f: (A, B) => C): A => (B => C) = (a: A) => (b: B) => f(a, b)

  def main(args: Array[String]): Unit = {
    val q: Int => Int => Int = curry[Int, Int, Int](_ + _)
    val r: Int = q(1)(2)
    println(r)

    val s = curry[Int, String, Double](_ * 3.14 + _.length)
    val t = s(2)("hello") // 2 * 3.14 + 5
    println((t, 2 * 3.14 + 5))
  }

}
