package com.klibisz.redbook

object Ex4_Options {

  sealed trait Option[+A] {
    def map[B](f: A => B): Option[B]
    def flatMap[B](f: A => Option[B]): Option[B]
    def getOrElse[B >: A](default: => B): B
    def orElse[B >: A](ob: => Option[B]): Option[B]
    def filter(f: A => Boolean): Option[A]
  }
  case class Some[+A](get: A) extends Option[A] {
    override def map[B](f: A => B): Option[B] = Some(f(get))
    override def flatMap[B](f: A => Option[B]): Option[B] = f(get)
    override def getOrElse[B >: A](default: => B): B = get
    override def orElse[B >: A](ob: => Option[B]): Option[B] = Some(get)
    override def filter(f: A => Boolean): Option[A] = if (f(get)) this else None
  }
  case object None extends Option[Nothing] {
    override def map[B](f: Nothing => B): Option[B] = None
    override def flatMap[B](f: Nothing => Option[B]): Option[B] = None
    override def getOrElse[B >: Nothing](default: => B): B = default
    override def orElse[B >: Nothing](ob: => Option[B]): Option[B] = ob
    override def filter(f: Nothing => Boolean): Option[Nothing] = None
  }
  object Option {
    def map2[A, B, C](oa: Option[A], ob: Option[B])(f: (A, B) => C): Option[C] =
      (oa, ob) match {
        case (Some(a), Some(b)) => Some(f(a, b))
        case _ => None
      }

    def sequence[A](la: List[Option[A]]): Option[List[A]] = traverse(la)(identity)

    /** Apply f to every member of a. Stop and return None if any application of f returns None. */
    def traverse[A, B](la: List[A])(f: A => Option[B]): Option[List[B]] = {
      @scala.annotation.tailrec
      def iter(la: List[A], acc: Vector[B]): Option[List[B]] =
        if (la.isEmpty) {
          if (acc.isEmpty) None
          else Some(acc.toList)
        } else f(la.head) match {
          case Some(b) => iter(la.tail, acc.appended(b))
          case None => None
        }
      iter(la, Vector.empty)
    }
  }

  def mean(xs: Seq[Double]): Option[Double] =
    if (xs.isEmpty) None
    else Some(xs.sum / xs.length)

  def variance(xs: Seq[Double]): Option[Double] =
    mean(xs).flatMap { m =>
      mean(xs.map(x => math.pow(x - m, 2)))
    }

  def main(args: Array[String]): Unit = {
    Seq[Option[Int]](Some(1), None).foreach { o =>
      println(o.map(_ + 1))
      println(o.flatMap(i => Some(i + 1)))
      println(o.getOrElse(2))
      println(o.orElse(Some(2)))
      println(o.filter(_ < 2))
      println("-" * 50)
    }

    println(mean(Seq(1,2,3)))
    println(variance(Seq(1,2,3)))

    println(Option.map2(Some(1), Some(-1))(math.max))
    println(Option.map2(Some(1), None)(math.max))
    println(Option.sequence(List(Some(1), Some(2))))
    println(Option.sequence(List(Some(1), Some(2), None)))

    println(Option.traverse(List(1,2,3))(n => if (n % 2 == 0) Some(n) else None))
    println(Option.traverse(List(2,4,6))(n => if (n % 2 == 0) Some(n) else None))
  }

}
