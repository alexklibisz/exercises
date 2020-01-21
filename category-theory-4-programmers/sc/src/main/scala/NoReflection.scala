
object NoReflection extends App {

  // In the client project, which knows nothing about the compiler.
  sealed trait Spec
  case class FooSpec(n: Int) extends Spec
  case class BarSpec(s: String) extends Spec

  // Everything below is in the "core" project, which depends on the client.
  case class InternalRepresentation(bleh: Double = Math.PI)

  // Typeclass representing a spec which can be used to transform an internal representation.
  trait Transforms[S <: Spec] {

    // This transformation takes an IR and the spec and returns an updated IR.
    def apply(spec: S, ir: InternalRepresentation): InternalRepresentation

    // This one takes just the spec and returns a function which will transform the IR.
    // This would let you build up a list of transformations (i.e. in parallel), and then apply them sequentially.
    // Maybe useful for slow, parallelizable transformations, like those which load data.
    def partial(spec: S): InternalRepresentation => InternalRepresentation
  }

  // Concrete implementations for each type of spec.
  object Transforms {
    implicit val foo: Transforms[FooSpec] = new Transforms[FooSpec] {
      def apply(foo: FooSpec, ir: InternalRepresentation): InternalRepresentation = ir.copy(bleh = ir.bleh + foo.n)
      def partial(spec: FooSpec): InternalRepresentation => InternalRepresentation = (ir: InternalRepresentation) => apply(spec, ir)
    }
  }

  // Over-simplified transformation function. The key thing is that any Spec passed to this function has to have
  // an instance of Transforms. As long as it does, the transformation logic will get looked up implicitly. This
  // saves us having to do reflection keep a huge mapping, etc.
  def transform[S <: Spec](specs: Seq[S])(implicit transforms: Transforms[S]): InternalRepresentation =
    specs.foldLeft(InternalRepresentation()) {
      case (acc, spec) => transforms(spec, acc)
    }

  val ir: InternalRepresentation = transform(Seq(FooSpec(1), FooSpec(2)))
  println(ir)

  // This won't compile until you provide an implementation of Transforms[BarSpec]
//  transform(Seq(FooSpec(1), BarSpec("Blah")))

  // TODO: figure out if there's a way to guarantee that there exists a Transforms[S] for all possible Specs.

}
