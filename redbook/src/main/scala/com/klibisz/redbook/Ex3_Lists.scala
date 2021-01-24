package com.klibisz.redbook

object Ex3_Lists {

  sealed trait List[+A]
  case object Nil extends List[Nothing]
  case class Cons[+A](head: A, tail: List[A]) extends List[A]

  object List {
    def tail[A](l: List[A]): List[A] = l match {
      case Nil => throw new UnsupportedOperationException("Called tail on an empty list.")
      case Cons(_, tail) => tail
    }
    def setHead[A](l: List[A], head: A): List[A] = l match {
      case Nil => l
      case Cons(_, tail) => Cons(head, tail)
    }
    def drop[A](l: List[A], n: Int): List[A] = {
      def iter(l: List[A], m: Int): List[A] =
        if (m == n) l
        else iter(tail(l), m + 1)
      iter(l, 0)
    }

    @scala.annotation.tailrec
    def dropWhile[A](l: List[A], f: A => Boolean): List[A] =
      l match {
        case Cons(head, tail) if f(head) => dropWhile(tail, f)
        case _ => l
      }

    def init[A](l: List[A]): List[A] =
      l match {
        case Nil => throw new UnsupportedOperationException("Called init on an empty list")
        case Cons(_, Nil) => Nil // At the end of the list, leave out the last element.
        case Cons(h, t) => Cons(h, init(t))
      }

    def foldRight[A, B](as: List[A], z: B)(f: (A, B) => B): B =
      as match {
        case Nil => z
        case Cons(x, xs) => f(x, foldRight(xs, z)(f))
      }

    @scala.annotation.tailrec
    def foldLeft[A, B](as: List[A], z: B)(f: (B, A) => B): B =
      as match {
        case Nil => z
        case Cons(h, tail) => foldLeft(tail, f(z, h))(f)
      }

    /** foldLeft in terms of foldRight. */
    def foldLeftViaRight[A, B](as: List[A], z: B)(f: (B, A) => B): B =
      foldRight(as, (b:B) => b)((a, g) => b => g(f(b, a)))(z)

    def foldRightViaFoldLeft[A, B](as: List[A], z:B)(f: (A, B) => B): B =
      foldLeft(reverse(as), z)((b,a) => f(a,b))

    def append[A](as: List[A], bs: List[A]): List[A] =
      foldLeft(reverse(as), bs)((tail, head) => Cons(head, tail))

    def sum(ns: List[Int]): Int = foldRight(ns, 0)(_ + _)

    def sumLeft(ns: List[Int]): Int = foldLeft(ns, 0)(_ + _)

    def product(ns: List[Double]): Double = foldRight(ns, 1d)(_ * _)

    def productLeft(ns: List[Double]): Double = foldLeft(ns, 1d)(_ * _)

    def apply[A](as: A*): List[A] = if (as.isEmpty) Nil else Cons(as.head, List(as.tail: _*))

    def length[A](as: List[A]): Int = foldRight(as, 0)((_, l) => l + 1)

    def lengthLeft[A](as: List[A]): Int = foldLeft(as, 0)((l, _) => l + 1)

    def reverse[A](as: List[A]): List[A] = foldLeft(as, Nil: List[A])((b, a) => Cons(a, b))

    def concat[A](lla: List[List[A]]): List[A] = {
      foldLeft(lla, Nil:List[A])(append)
    }

    def map[A, B](la: List[A])(f: A => B): List[B] = foldLeft(reverse(la), Nil: List[B])((lb, a) => Cons(f(a), lb))

    def filter[A](la: List[A])(f: A => Boolean): List[A] =
      foldRightViaFoldLeft(la, Nil:List[A])((a, acc) => if (f(a)) Cons(a, acc) else acc)

    def flatMap[A, B](la: List[A])(f: A => List[B]): List[B] = concat(map(la)(f))

    def filterViaFlatMap[A](la: List[A])(f: A => Boolean): List[A] =
      flatMap(la)(a => if (f(a)) Cons(a, Nil) else Nil)

    def zipWith[A, B, C](la: List[A], lb: List[B])(f: (A, B) => C): List[C] = {
      def iter(la: List[A], lb: List[B], lc: List[C]): List[C] = {
        (la, lb) match {
          case (_, Nil) => reverse(lc)
          case (Nil, _) => reverse(lc)
          case (Cons(ha, ta), Cons(hb, tb)) => iter(ta, tb, Cons(f(ha, hb), lc))
        }
      }
      iter(la, lb, Nil: List[C])
    }

    def take[A](la: List[A], n: Int): List[A] = {
      @scala.annotation.tailrec
      def iter(la: List[A], m: Int, acc: List[A]): List[A] = {
        if (m == n) reverse(acc)
        else la match {
          case Nil => reverse(acc)
          case Cons(head, tail) => iter(tail, m + 1, Cons(head, acc))
        }
      }
      iter(la, 0, Nil: List[A])
    }

    def hasSubSequence[A](sup: List[A], sub: List[A]): Boolean = {
      val l = length(sub)
      @scala.annotation.tailrec
      def iter(la: List[A]): Boolean = {
        if (la == Nil) false
        else if (take(la, l) == sub) true
        else iter(tail(la))
      }
      iter(sup)
    }

  }

  def main(args: Array[String]): Unit = {
    val l: List[Int] = Cons(1, Cons(2, Nil))
    println(List.tail(l)) // Cons(2, Nil)
    println(List.setHead(l, 2)) // Cons(2, Cons(2, Nil))
    println(List.drop(l, 2)) // Nil
    println(List.dropWhile(l, (i: Int) => i < 2)) // Cons(2, Nil)
    println(List.init(Cons(1, Cons(2, Cons(3, Nil))))) // Cons(1, Cons(2, Nil))
    println(List.foldRight(List(1,2,3), Nil: List[Int])(Cons(_, _)))
    println(List.length(List(1,2,3,4)))
    println(List.foldLeft(List(1,2,3), Nil: List[Int])((b, a) => Cons(a, b)))
    println(List.sumLeft(List(1,2,3)))
    println(List.productLeft(List(1,2,3)))
    println(List.lengthLeft(List(1,2,3)))
    println(List.reverse(List(1,2,3)))
    println(List.append(List(1,2,3), List(4,5,6)))
    println(List.concat(List(List(1,2,3), List(4,5,6,7), List(8, 9))))
    println(List.map(List(1,2,3))(_ + 1))
    println(List.map(List(1.1,2.2,3.3))(_.toString))
    println(List.filter(List(1,2,3,4,5,6))(_ % 2 == 0))
    println(List.flatMap(List(1,2,3))(i => List(i, i)))
    println(List.filterViaFlatMap(List(1,2,3,4,5,6))(_ % 2 == 0))
    println(List.zipWith(List(1,2,3), List(4,5,6))(_ + _))
    println(List.take(List(1,2,3,4,5), 3))
    println(List.hasSubSequence(List(1,2,3,4,5,6), List(3,4,5)))
    println(List.hasSubSequence(List(1,2,3,4,5,6), List(3,4,4)))
  }


}
