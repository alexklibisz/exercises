package com.klibisz.redbook

object Ex3_Trees {

  sealed trait Tree[+A]
  case class Leaf[A](value: A) extends Tree[A]
  case class Branch[A](left: Tree[A], right: Tree[A]) extends Tree[A]

  object Tree {
    def size[A](t: Tree[A]): Int = t match {
      case Leaf(_) => 1
      case Branch(l, r) => Tree.size(l) + Tree.size(r)
    }

    def max[A : Ordering](t: Tree[A]): A =
      t match {
        case Leaf(n) => n
        case Branch(l, r) => implicitly[Ordering[A]].max(max(l), max(r))
      }

    def min[A: Ordering](t: Tree[A]): A =
      t match {
        case Leaf(n) => n
        case Branch(l, r) => implicitly[Ordering[A]].min(min(l), min(r))
      }

    def depth[A](t: Tree[A]): Int = {
      def d(t: Tree[A], depth: Int): Int = t match {
        case Leaf(_) => depth
        case Branch(l, r) => d(l, depth + 1) max d(r, depth + 1)
      }
      d(t, 0)
    }

    def map[A, B](t: Tree[A])(f: A => B): Tree[B] = t match {
      case Leaf(a) => Leaf(f(a))
      case Branch(l, r) => Branch(map(l)(f), map(r)(f))
    }

  }

  def main(args: Array[String]): Unit = {
    val t1 = Branch(Branch(Leaf(1), Leaf(2)), Branch(Leaf(3), Branch(Leaf(-1), Leaf(4))))
    println(Tree.size(t1))
    println(Tree.max(t1))
    println(Tree.min(t1))
    println(Tree.depth(t1))
    println(Tree.map(t1)(_ + 1))
  }

}
