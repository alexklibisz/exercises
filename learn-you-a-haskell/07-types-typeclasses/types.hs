-- Shape is a type with two type constructors: Circle and Rectangle.
-- Circle and Rectangle are not types themselves.
-- You couldn't write a function that works only for Circles.
-- Anything that takes a Shape *should* be implemented for all type constructors.
-- Use `deriving (Show)` to automagically make shapes printable.
-- Point left of the equals is the name of the data type. Point on the right is the name of the value constructor.
--  It could also be called MkPoint, Foo, Bar, etc. It's really just a function that takes some values and returns
--  an instance of a shape. I think that helps explain why Circle and Rectangle are not types themselves, rather 
--  type constructors for the Shape type.
-- Deriving Show and Eq means that haskell will automagically make these types implement the Eq and Show typeclasses.
data Point = Point Float Float deriving (Show, Eq)
data Shape = Circle Point Float | Rectangle Point Point deriving (Show, Eq)

-- Example: surfaceArea $ Circle 0 0 3
surfaceArea :: Shape -> Float
surfaceArea (Circle _ r) = pi * r ^ 2
surfaceArea (Rectangle (Point x1 y1) (Point x2 y2)) = (abs $ x2 - x1) * (abs $ y2 - y1)

-- Record syntax is basically a case class, especially when using deriving (Show, Eq).
-- Again, Car on the LHS and Car on the RHS don't have to be exactly the same, but it's convention to do so.
data Car = Car {company :: String, model :: String, year :: Int} deriving (Show, Eq)

-- My first car.
camry1997 = Car {model = "Camry", company = "Toyota", year = 1997}

-- Type constructors take a (lowercase) type and construct another (uppercase) type from it.
-- Translating Scala Option (even though Maybe already exists and is equivalent).
data Option a = None | Some a

-- Example using a typeclass. Never put the typeclass constraint in the data constructor.
-- You'll have to put it in the functions anyways, and you don't want to do that for functions
-- that don't require the constraint.
-- Again, type constructor on the left, value constructor on the right (of the equal sign).
data Vector a = Vector a a a deriving (Show)

-- Define infix functions for adding, multiplying, dotting two vectors.
vplus :: (Num t) => Vector t -> Vector t -> Vector t
(Vector i j k) `vplus` (Vector l m n) = Vector (i+l) (j+m) (k+n)

vmul :: (Num t) => Vector t -> t -> Vector t
(Vector i j k) `vmul` m = Vector (i * m) (j * m) (k * m)

vdot :: (Num t) => Vector t -> Vector t -> t
(Vector i j k) `vdot` (Vector l m n) = i * l + j * m + k * n

-- If you ain't first you're last. To ensure First > Last with deriving (Ord), you have to 
-- declare the Last data constructor first.. hmmm....
-- Example: 
-- *Main> First > Last
-- True
-- *Main> First < Last
-- False
-- *Main> First == Last
-- False
-- Contrary to the book, you also have to derive Eq.
data FirstOrLast = Last | First deriving (Eq, Ord)


-- Another, possibly more useful, example is to define orderable enums.
-- Intuitively, deriving Enum requires that all value constructors take no parameters.
data Day = Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday   
           deriving (Eq, Ord, Show, Read, Bounded, Enum)

-- Recursive data structures.
infixr 5 :-:
data List a = Empty | Cons a (List a) | a :-: (List a) deriving (Show, Read, Eq, Ord)

infixr 5 .++ 
(.++) :: List a -> List a -> List a
Empty .++ ys = ys
(x :-: xs) .++ ys = x :-: (xs .++ ys)
