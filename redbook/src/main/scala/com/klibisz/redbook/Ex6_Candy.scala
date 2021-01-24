package com.klibisz.redbook

object Ex6_Candy {

  /** Computation that carries some state along. */
  final case class State[S, +A](run: S => (A, S)) {
    def apply(s: S): (A, S) = run(s)

    def map[B](f: A => B): State[S, B] = flatMap(a => State.unit(f(a)))

    def map2[B, C](sb: State[S, B])(f: (A, B) => C): State[S, C]  =
      flatMap(a => sb.map(b => f(a, b)))

    def flatMap[B](f: A => State[S, B]): State[S, B] = State { s =>
      val (a, s1) = run(s)
      f(a).run(s1)
    }

  }

  object State {
    def unit[S, A](a: A): State[S, A] = State(s => (a, s))
    def sequence[S, A](sas: List[State[S, A]]): State[S, List[A]] = {
      def go(s: S, actions: List[State[S,A]], acc: List[A]): (List[A],S) =
        actions match {
          case Nil => (acc.reverse,s)
          case h :: t => h.run(s) match { case (a,s2) => go(s2, t, a :: acc) }
        }
      State((s: S) => go(s,sas,List()))
    }
    def get[S]: State[S, S] = State(s => (s, s))
    def set[S](s: S): State[S, Unit] = State(_ => ((), s))
    def getAndSet[S](f: S => S): State[S, Unit] = for {
      s <- get
      _ <- set(f(s))
    } yield ()
  }

  sealed trait Input
  case object Coin extends Input
  case object Turn extends Input

  // Insert coin into locked machine -> unlock if there's any candy left.
  // Turn know on unlocked machine -> dispenses candy and becomes locked.
  // Turn knob on locked machine -> nothing
  // Insert coin into unlocked machine -> nothing
  // Any input to machine with no candy -> nothing
  case class Machine(locked: Boolean, candies: Int, coins: Int)

  object Machine {

    import State._

    def update(input: Input)(curr: Machine): Machine =
      (input, curr) match {
        case (_, Machine(_, 0, _)) => curr
        case (Coin, Machine(false, _, _)) => curr
        case (Turn, Machine(true, _, _)) => curr
        case (Coin, Machine(true, candy, coin)) => Machine(false, candy, coin + 1)
        case (Turn, Machine(false, candy, coin)) => Machine(true, candy - 1, coin)
      }

    /** Returns a state which takes a machine, applies the inputs, produces the resulting machine. */
    def simulate(inputs: List[Input]): State[Machine, Machine] =
      for {
        _ <- sequence(inputs.map(getAndSet[Machine] _ compose update))
        r <- get
      } yield r


  }

  def main(args: Array[String]): Unit = {
    val m0 = Machine(true, 5, 10)
    val s = Machine.simulate(List(Coin, Turn, Coin, Turn))
    val m1: (Machine, Machine) = s(m0)
    println((m0, m1))
  }


}
