package com.klibisz.redbook

object Ex6_RNGCombinators {

  type Rand[+A] = RNG => (A, RNG)

  trait RNG {
    def nextInt: (Int, RNG)
  }

  final class SimpleRNG(val seed: Long) extends RNG {
    def nextInt: (Int, RNG) = {
      val s2 = (seed * 0x5DEECE66DL + 0xBL) & 0xFFFFFFFFFFFFL
      val rng2 = new SimpleRNG(s2)
      val i1 = (s2 >>> 16).toInt
      i1 -> rng2
    }
  }

  object RNG {
    def unit[A](a: A): Rand[A] = rng => (a, rng)
    def map[A, B](s: Rand[A])(f: A => B): Rand[B] = rng => {
      val (a, rng2) = s(rng)
      (f(a), rng2)
    }
    def int: Rand[Int] = _.nextInt
    def double: Rand[Double] = map(int)(_ * 1d / Int.MaxValue)
    def nonNegativeInt: Rand[Int] = { rng =>
      val (i, rng2) = rng.nextInt
      (i + 1).abs -> rng2
    }
    def nonNegativeEven: Rand[Int] =
      map(nonNegativeInt)(i => i - i % 2)

    def map2[A, B, C](ra: Rand[A], rb: Rand[B])(f: (A, B) => C): Rand[C] = { rng =>
      val (a, rnga) = ra(rng)
      val (b, rngb) = rb(rnga)
      f(a, b) -> rngb
    }

    def both[A, B](ra: Rand[A], rb: Rand[B]): Rand[(A, B)] =
      map2(ra, rb)(Tuple2(_, _))

    def randIntDouble: Rand[(Int, Double)] = both(int, double)

    def randDoubleInt: Rand[(Double, Int)] = both(double, int)

    /** Combine a list of transitions into a single transition. */
    def sequence[A](fs: List[Rand[A]]): Rand[List[A]] = { rng =>
      val (va, rng2) = fs.foldLeft((Vector.empty[A], rng)) {
        case ((acc: Vector[A], rng: RNG), ra: Rand[A]) =>
          val (a, rng2) = ra(rng)
          (acc :+ a) -> rng2
      }
      va.toList -> rng2
    }

    def sequence2[A](fs: List[Rand[A]]): Rand[List[A]] = rng =>
      fs.foldRight((List.empty[A], rng)) {
        case (ra: Rand[A], (acc: List[A], rng: RNG)) =>
          val (a, rng2) = ra(rng)
          (a +: acc) -> rng2
      }

    def ints[A](count: Int): Rand[List[Int]] = sequence(List.fill(count)(int))

    def flatMap[A, B](f: Rand[A])(g: A => Rand[B]): Rand[B] = { rng =>
      val (a, rng2) = f(rng)
      g(a)(rng2)
    }

    def nonNegativeLessThan(n: Int): Rand[Int] = flatMap(nonNegativeInt) { i =>
      val mod = i % n
      if (i + (n -1) - mod >= 0) unit(mod) else nonNegativeLessThan(n)
    }

    def mapViaFlatMap[A, B](s: Rand[A])(f: A => B): Rand[B] = flatMap(s)(a => unit(f(a)))

    def map2ViaFlatMap[A, B, C](ra: Rand[A], rb: Rand[B])(f: (A, B) => C): Rand[C] =
      flatMap(ra)(a => map(rb)(b => f(a, b)))

  }

  def main(args: Array[String]): Unit = {

    val rng1 = new SimpleRNG(0)
    val (i1, rng2) = RNG.nonNegativeInt(rng1)
    val (i2, rng3) = RNG.nonNegativeEven(rng2)
    val (d1, _) = RNG.double(rng3)

    println((i1, i2, d1))
    println(RNG.ints(10)(rng3)._1)
  }

}
