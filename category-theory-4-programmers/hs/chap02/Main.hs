import Data.Void -- I really really wish they would include compilable snippets (i.e. including imports)
                 -- in these books so you don't have to waste time fumbling for how to actually run the program.

fact n = product [1..n]

absurd :: Void -> a
absurd = undefined

-- Unit () is a dummy value of which there is only one instance ever.
f44 :: () -> Integer
f44 () = 44

-- There exists exactly one function which maps type A to () for every type A.
fInt :: Integer -> ()
fInt _ = ()

-- Functions that can be implemented with the same "formula" for any type are called parametrically polymorphic. 
unit :: a -> ()
unit _ = ()

-- The set Bool can be defined:
data Bool = True | False

main = do
    print (fact 10)
    print (f44 ())
    print (fInt 22)
    print (unit "hi")
    print (unit 33)