package com.klibisz.redbook

object Ex4_Either {

  sealed trait Either[+E, +A] {
    final def map[B](f: A => B): Either[E, B] = this match {
      case Left(value) => Left(value)
      case Right(value) => Right(f(value))
    }
    def flatMap[EE >: E, B](f: A => Either[EE, B]): Either[EE, B] = this match {
      case Left(value) => Left(value)
      case Right(value) => f(value)
    }
    def orElse[EE >: E, B >: A](b: => Either[EE, B]): Either[EE, B] = this match {
      case Left(_) => b
      case Right(value) => Right(value)
    }
    def map2[EE >: E, B, C](b: Either[EE, B])(f: (A, B) => C): Either[EE, C] = (this, b) match {
      case (Right(a), Right(b)) => Right(f(a, b))
      case (Left(e), _) => Left[EE](e)
      case (_, Left(e)) => Left[EE](e)
    }
    def isLeft: Boolean = this match {
      case Left(_) => true
      case Right(_) => false
    }
    def isRight: Boolean = !isLeft
  }

  case class Left[+E](value: E) extends Either[E, Nothing]
  case class Right[+A](value: A) extends Either[Nothing, A]

  object Either {
    def sequence[E, A](es: List[Either[E, A]]): Either[E, List[A]] = traverse(es)(identity)

    def traverse[E, A, B](as: List[A])(f: A => Either[E, B]): Either[E, List[B]] = {
      @scala.annotation.tailrec
      def iter(as: List[A], acc: Vector[B]): Either[E, List[B]] =
        if (as.isEmpty) Right(acc.toList)
        else f(as.head) match {
          case Left(value) => Left(value)
          case Right(value) => iter(as.tail, acc.appended(value))
        }
      iter(as, Vector.empty)
    }
  }

  case class Person(name: Name, age: Age)
  sealed class Name(val value: String)
  sealed class Age(val value: Int)

  def mkName(name: String): Either[String, Name] =
    if (name == "" || name == null) Left("Name is empty.")
    else Right(new Name(name))

  def mkAge(age: Int): Either[String, Age] =
    if (age < 0) Left("Age is invalid.")
    else Right(new Age(age))

  def mkPerson(name: String, age: Int): Either[String, Person] =
    mkName(name).map2(mkAge(age))(Person.apply)

  def mkPerson2(name: String, age: Int): Either[List[String], Person] =
    mkName(name) -> mkAge(age) match {
      case (Right(name), Right(age)) => Right(Person(name, age))
      case (Left(s1), Left(s2)) => Left(List(s1, s2))
      case (_, Left(s2)) => Left(List(s2))
      case (Left(s1), _) => Left(List(s1))
    }


  def main(args: Array[String]): Unit = {
    val l1: Either[Exception, Int] = Left(new Exception("l1"))
    val l2: Either[Exception, Int] = Left(new Exception("l2"))
    val r1: Either[Exception, Int] = Right(42)
    val r2: Either[Exception, Int] = Right(43)
    println(l1.map2(l2)((a, b) => a + b))
    println(r1.map2(l1)((a, b) => a + b))
    println(r1.map2(r2)((a, b) => a + b))
    println(Either.sequence(List(r1, r2)))
    println(Either.sequence(List(r1, l1, l2)))

    println(mkPerson2("", -1))
  }

}
