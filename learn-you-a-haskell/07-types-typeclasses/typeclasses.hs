-- Examples of implementing typeclasses for the Option type.
data Option a = None | Some a

-- To implement Eq, you have to constrain that the type `a` also implements Eq.
-- *Main> (Some 1) == (Some 2)
-- False
-- *Main> (Some 1) == (Some 1)
-- True
instance (Eq a) => Eq (Option a) where
    Some x == Some y = x == y
    None == None = True
    _ == _ = False 

-- Same with Show
instance (Show a) => Show (Option a) where
    show (Some a) = "Some(" ++ (show a) ++ ")" 
    show None = "None"

-- True-ish typeclass
class Trueish a where
    trueish :: a -> Bool

instance Trueish Int where
    trueish 0 = False
    trueish _ = True

instance Trueish [a] where
    trueish [] = False
    trueish _ = True

instance Trueish Bool where
    trueish = id -- id is a stdlib function that takes a parameter and returns the same thing.

instance Trueish (Maybe a) where
    trueish (Just _) = True
    trueish Nothing = False

-- If-like function that operates on trueish values.
ifish :: (Trueish a) => a -> b -> b -> b
ifish pred yesish noish = if trueish pred then yesish else noish 