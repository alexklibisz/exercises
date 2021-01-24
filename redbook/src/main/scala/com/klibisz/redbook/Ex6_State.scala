package com.klibisz.redbook

import com.klibisz.redbook.Ex6_State.State.unit

object Ex6_State {

  /** Computation that carries some state along. */
  final case class State[S, +A](run: S => (A, S)) {
    def apply(s: S): (A, S) = run(s)

    def map[B](f: A => B): State[S, B] = flatMap(a => unit(f(a)))

    def map2[B, C](sb: State[S, B])(f: (A, B) => C): State[S, C]  =
      flatMap(a => sb.map(b => f(a, b)))

    def flatMap[B](f: A => State[S, B]): State[S, B] = State { s =>
      val (a, s1) = run(s)
      f(a).run(s1)
    }
  }

  object State {
    def unit[S, A](a: A): State[S, A] = State(s => (a, s))
    def sequence[S, A](ss: List[State[S, A]]): State[S, List[A]] = State { state =>
      ss.foldRight((List.empty[A], state)) {
        case (state: State[S, A], (acc: List[A], s0)) =>
          val (a, s1) = state(s0)
          println(acc)
          (a +: acc) -> s1
      }
    }
  }

  def main(args: Array[String]): Unit = {
    val square = State[Int, Double](i => (math.pow(i, 2), i + 1))
    val _ = {
      val (i1, s1) = square(0)
      val (i2, s2) = square(s1)
      val (i3, s3) = square(s2)
      println((i1, i2, i3))
    }

    val squareAndDouble = square.map(_ * 2)
    val _ = {
      val (i1, s1) = squareAndDouble(0)
      val (i2, s2) = squareAndDouble(s1)
      val (i3, s3) = squareAndDouble(s2)
      println((i1, i2, i3))
    }

    val squaresAndDoubles: List[State[Int, Double]] = List.fill(10)(squareAndDouble)
    val squaresAndDoublesSeq = State.sequence(squaresAndDoubles)
    println(squaresAndDoublesSeq(0)._1)


  }


}
