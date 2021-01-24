package com.klibisz.redbook

object Ex2_5 {

  def compose[A, B, C](f: B => C, g: A => B): A => C = (a: A) => f(g(a))

  def main(args: Array[String]): Unit = {
    val f: String => Int = _.length
    val g: Double => String = _.floor.toInt.toString
    val h = compose(f, g)
    val q = h(69.420)
    println(q)
  }

}
