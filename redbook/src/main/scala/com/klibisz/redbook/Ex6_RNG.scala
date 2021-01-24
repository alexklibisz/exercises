package com.klibisz.redbook

object Ex6_RNG {

  trait RNG {
    def nextInt: (Int, RNG)
    def nonNegativeInt: (Int, RNG)
    def nextDouble: (Double, RNG)
    def intDouble: ((Int, Double), RNG)
    def doubleInt: ((Double, Int), RNG)
    def double3: ((Double, Double, Double), RNG)
    def ints(count: Int): (List[Int], RNG)
  }

  final class SimpleRNG(val seed: Long) extends RNG {
    override def nextInt: (Int, RNG) = {
      val s2 = (seed * 0x5DEECE66DL + 0xBL) & 0xFFFFFFFFFFFFL
      val rng2 = new SimpleRNG(s2)
      val i1 = (s2 >>> 16).toInt
      i1 -> rng2
    }
    override def nonNegativeInt: (Int, RNG) = {
      val (i, rng2) = nextInt
      (i + 1).abs -> rng2
    }
    override def nextDouble: (Double, RNG) = {
      val (i, rng2) = nextInt
      (i * 1.0 / Int.MaxValue) -> rng2
    }

    override def intDouble: ((Int, Double), RNG) = {
      val (i1, rng2) = nextInt
      val (d1, rng3) = rng2.nextDouble
      (i1, d1) -> rng3
    }

    override def doubleInt: ((Double, Int), RNG) = {
      val ((i1, d1), rng2) = intDouble
      (d1, i1) -> rng2
    }

    override def double3: ((Double, Double, Double), RNG) = {
      val (d1, rng2) = nextDouble
      val (d2, rng3) = rng2.nextDouble
      val (d3, rng4) = rng3.nextDouble
      (d1, d2, d3) -> rng4
    }

    override def ints(count: Int): (List[Int], RNG) = {
      (0 until count).foldLeft((List.empty[Int], this: RNG)) {
        case ((acc, rng), _) =>
          val (i, rng2) = rng.nextInt
          (i +: acc, rng2)
      }
    }
  }

  def main(args: Array[String]): Unit = {

    val rng1 = new SimpleRNG(0)
    val (i1, rng2) = rng1.nextInt
    val (i2, rng3) = rng2.nextInt
    val (i3, _) = rng3.nextInt
    println((i1, i2, i3))

    println(rng3.nonNegativeInt._1)
    println(rng3.nextDouble._1)
    println(rng3.intDouble._1)
    println(rng3.doubleInt._1)
    println(rng3.double3._1)
    println(rng3.ints(5)._1)



  }

}
