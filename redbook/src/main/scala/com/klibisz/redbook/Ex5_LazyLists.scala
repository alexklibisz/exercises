package com.klibisz.redbook

import com.klibisz.redbook.Ex5_LazyLists.Stream.{cons, empty, unfold}

import javax.crypto.ExemptionMechanism
import scala.annotation.tailrec

object Ex5_LazyLists {

  sealed trait Stream[+A] {
//    def headOption: Option[A] = this match {
//      case Empty => None
//      case Cons(h, _) => Some(h()) // Call the h thunk.
//    }

    def toList: List[A] = this match {
      case Empty => List.empty[A]
      case Cons(h, t) => List(h()) ++ t().toList
    }

    def take(n: Int): Stream[A] =
      this match {
        case Empty => Empty
        case Cons(h, t) =>
          if (n == 0) Empty
          else Stream.cons(h(), t().take(n - 1))
      }

    @scala.annotation.tailrec
    final def drop(n: Int): Stream[A] = if (n == 0) this else this match {
      case Empty => Empty
      case Cons(_, t) => t().drop(n - 1)
    }

//    def takeWhile(p: A => Boolean): Stream[A] =
//      this match {
//        case Empty => Empty
//        case Cons(h, t) => if (p(h())) Stream.cons(h(), t().takeWhile(p)) else Empty
//      }

    @scala.annotation.tailrec
    final def exists(p: A => Boolean): Boolean = this match {
      case Cons(h, t) => p(h()) || t().exists(p)
      case _ => false
    }

    // final def exists(p: A => Boolean): Boolean = foldRight(false)((a, b) => p(a) || b)

    final def foldRight[B](z: => B)(f: (A, => B) => B): B =
      this match {
        case Empty => z
        case Cons(h, t) => f(h(), t().foldRight(z)(f))
      }

    @scala.annotation.tailrec
    final def forAll(p: A => Boolean): Boolean =
      this match {
        case Empty => true
        case Cons(h, t) => if (p(h())) t().forAll(p) else false
      }

    final def takeWhile(p: A => Boolean): Stream[A] =
      foldRight(Empty: Stream[A]) { (a, acc) => if (p(a)) Stream.cons(a, acc) else Empty }

    final def headOption: Option[A] =
      foldRight(Option.empty[A])((a, _) => Some(a))

    final def map[B](f: A => B): Stream[B] =
      foldRight(Empty: Stream[B])((a, sb) => Stream.cons(f(a), sb))

    final def append[AA >: A](saa: => Stream[AA]): Stream[AA] = {
      lazy val memo = saa
      foldRight(memo)((a, tail) => Stream.cons(a, tail))
    }

    final def flatMap[B](f: A => Stream[B]): Stream[B] =
      foldRight(empty[B])((h, t) => f(h).append(t))

    final def filter(p: A => Boolean): Stream[A] = this match {
      case Empty => Empty
      case Cons(h, t) => if (p(h())) cons(h(), t().filter(p)) else t().filter(p)
    }

    final def find(p: A => Boolean): Option[A] = filter(p).headOption

    final def mapViaUnfold[B](f: A => B): Stream[B] = unfold(this) {
      case Cons(h, t) => Some((f(h()), t()))
      case Empty => None
    }

    final def takeViaUnfold(n: Int): Stream[A] = unfold((this, 0)) {
      case (Cons(h, t), m) if m < n => Some((h(), (t(), m + 1)))
      case _ => None
    }

    final def takeWhileViaUnfold(p: A => Boolean): Stream[A] = unfold(this) {
      case Cons(h, t) if p(h()) => Some((h(), t()))
      case _ => None
    }

    final def zipWithViaUnfold[B, C](sb: Stream[B])(f: (A, B) => C): Stream[C] =
      unfold((this, sb)) {
        case (Cons(ha, ta), Cons(hb, tb)) => Some(f(ha(), hb()) -> (ta(), tb()))
        case _ => None
      }

    final def zipAll[B](sb: Stream[B]): Stream[(Option[A], Option[B])] =
      unfold((this, sb)) {
        case (Cons(ha, ta), Cons(hb, tb)) => Some(((Some(ha()), Some(hb())), (ta(), tb())))
        case (Empty, Cons(hb, tb)) => Some(((None, Some(hb())), (Empty, tb())))
        case (Cons(ha, ta), Empty) => Some(((Some(ha()), None), (ta(), Empty)))
      }

    @scala.annotation.tailrec
    final def startsWith[B >: A](other: Stream[B]): Boolean = (this, other) match {
      case (Cons(ha, ta), Cons(hb, tb)) =>
        if (ha() == hb()) ta().startsWith(tb()) else false
      case (_, Empty) => true
      case _ => false
    }

    final def tails(): Stream[Stream[A]] = unfold(Option(this)) {
      case Some(Empty) => Some((Stream[A](), None))
      case Some(s @ Cons(_, t)) => Some((s, Some(t())))
      case _ => None
    }

    final def hasSubSequence[A](s: Stream[A]): Boolean = tails().exists(_.startsWith(s))

    final def scanRight[B](z: B)(f: (A, B) => B): Stream[B] = {
      foldRight(Stream(z)) {
        case (a, sb @ Cons(hb, _)) => cons(f(a, hb()), sb)
        case (_, b) => b
      }
    }

  }
  case object Empty extends Stream[Nothing]
  case class Cons[+A](h: () => A, t: () => Stream[A]) extends Stream[A]

  object Stream {
    /** Smart constructor caches head and tail to avoid re-computing them. */
    def cons[A](h: => A, t: => Stream[A]): Stream[A] = {
      lazy val lvh = h
      lazy val lvt = t
      Cons(() => lvh, () => lvt)
    }
    def empty[A]: Stream[A] = Empty
    def apply[A](as: A*): Stream[A] = if (as.isEmpty) empty else cons(as.head, apply(as.tail: _*))
    def ones(): Stream[Int] = Stream.cons(1, ones)
    def constant[A](a: A): Stream[A] = cons(a, constant(a))
    def from(a: Int): Stream[Int] = cons(a, from(a + 1))
    def fibs(): Stream[Int] = {
      def fibs(a: Int, b: Int): Stream[Int] = cons(a, fibs(b, a + b))
      fibs(0, 1)
    }
    /** Takes an initial state and a function for producing both the next state and the next value. */
    def unfold[A, S](z: S)(f: S => Option[(A, S)]): Stream[A] =
      f(z) match {
        case Some((a, s)) => cons(a, unfold(s)(f))
        case None => Empty
      }

    def fibsViaUnfold(): Stream[Int] =
      unfold((0,1)) {
        case (a, b) => Some(a, (b, a + b))
      }

    def fromViaUnfold(a: Int): Stream[Int] = unfold(a)(n => Some(n, n + 1))

    def constantViaUnfold[A](a: A): Stream[A] = unfold(a)(_ => Some((a, a)))

    def onesViaUnfold(): Stream[Int] = unfold(1)(_ => Some((1, 1)))

  }

  def main(args: Array[String]): Unit = {
    // Make sure the cons constructor memoizes.
    val s1 = Stream.cons({ println("computing head..."); 1 }, Empty)
    println(s1.headOption) // "computing head..." printed once here.
    println(s1.headOption)

    val s2 = Cons(() => { println("computing head..."); 2 }, () => Empty)
    println(s2.headOption) // "computing head..." printed once here.
    println(s2.headOption) // "computing head..." printed again here.

    println(Stream(1,2,3).toList)
    println(Stream(1, 2, 3, 4, 5, 6).take(2).toList)
    println(Stream(1,2,3,4,5).drop(2).toList)
    println(Stream(1,2,3,4,5).takeWhile(_ < 4).toList)
    println(Stream(1,2,3).exists(_ == 3))
    println(Stream(1,2,3,4,5).forAll(_ < 10))
    println(Stream(1,2,3,4).forAll(_ < 3))
    println(Stream(1,2,3,4,5).takeWhile(_ < 4).toList)

    println(Stream(1,2,3,4,5).headOption)

    println(Stream(1,2,3,4,5).map(_ + 1).toList)
    println(Stream(1,2,3,4).append(Stream(5,6,7,8)).toList)

    println(Stream(1,2,3,4).flatMap(i => Stream(i, -i)).toList)

    println(Stream(1,2,3,4).find(_ > 3))
    println(Stream(1,2,3,4).find(_ < 0))


    println(Stream.constant(22).take(3).toList)

    println(Stream.from(2).take(3).toList)

    println(Stream.fibs().take(6).toList)

    println(Stream.fibsViaUnfold().take(6).toList)
    println(Stream.fromViaUnfold(2).take(3).toList)

    println(Stream.constantViaUnfold("bob").take(3).toList)
    println(Stream.ones().take(4).toList)

    println(Stream(1,2,3).mapViaUnfold(_ * 2).toList)
    println(Stream.from(2).takeViaUnfold(5).toList)
    println(Stream.from(2).takeWhileViaUnfold(_ < 10).toList)

    println(Stream.from(2).zipWithViaUnfold(Stream.from(3))(Tuple2(_, _)).take(3).toList)

    println(Stream.from(2).zipAll(Stream.from(2).take(3)).take(8).toList)

    println(Stream.from(2).startsWith(Stream(2,3,4)))
    println(Stream.from(2).startsWith(Stream(1,2,3)))
    println(Stream.from(2).startsWith(Stream.empty))

    println(Stream.from(1).take(3).tails().map(_.toList).toList)

    println(Stream.from(1).take(99).hasSubSequence(Stream.from(88).take(10)))

    println(Stream(1,2,3).scanRight(0)(_ + _).toList)

  }
}
