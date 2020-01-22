{-# language FlexibleInstances #-}
{-# language MultiParamTypeClasses #-}
{-# language FunctionalDependencies #-}


-- Typeclass whose instances are categories. Cateogires have objects and morphisms.
-- Morphisms each have an input object, the domain, and an output object, the codomain.
class Category obj mor | mor -> obj where
    -- Given a morphism you can get its input object.
    dom :: mor -> obj
    -- Given a morphism you can get its output object.
    cod :: mor -> obj
    -- Given an object you can get its identity morphism.
    idy :: obj -> mor
    -- Two morphisms might not compose: f: a -> b and g: b' -> c compose iff b = b'.
    cmp :: mor -> mor -> Maybe mor

data Object = One | Two deriving (Show, Eq)
data Morphism = Morphism Object Object deriving (Show, Eq)

instance Category Object Morphism where
    dom (Morphism a b) = a
    cod (Morphism a b) = b
    idy a = Morphism a a
    cmp (Morphism One One) (Morphism One Two) = Just (Morphism One Two)
    cmp (Morphism One Two) (Morphism Two Two) = Just (Morphism Two Two)
    cmp m1 m2
        | m1 == m2 = Just m1
        | otherwise = Nothing

main = do
    print (dom (Morphism One Two)) -- "One"
    print (cod (Morphism One One)) -- "One"
    print (cod (Morphism Two Two)) -- "Two"
    -- TODO: why do you have to have type annotation for idy?
    print (idy One :: Morphism) -- "Morphism One One" 
    print (idy Two :: Morphism) -- "Morphism Two Two"
    print (cmp (Morphism One One) (Morphism One Two)) -- "Just (Morphism One Two)"
    print (cmp (Morphism One One) (Morphism Two Two)) -- "Nothing"