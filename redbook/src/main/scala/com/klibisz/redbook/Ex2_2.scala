package com.klibisz.redbook

object Ex2_2 {

  def isSorted[A](as: Array[A], ordered: (A,A) => Boolean): Boolean = {
    @scala.annotation.tailrec
    def iter(i: Int): Boolean =
      if (i == as.length - 1) true
      else if (ordered(as(i), as(i + 1))) iter(i + 1)
      else false
    iter(0)
  }

  def main(args: Array[String]): Unit = {
    println(isSorted[Int](Array(1,2,3,4), (a, b) => a <= b))
    println(isSorted[Int](Array(1,2,4,3), (a, b) => a <= b))
  }

}
