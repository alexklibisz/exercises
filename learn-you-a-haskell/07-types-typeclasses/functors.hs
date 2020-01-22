-- "f is not a a concrete type, but a type constructor that takes one type as the parameter"
-- How do you know this? Because of the shape f a and f b below, showing that f gets applied
-- to a single a and a single b
class Functor' f where
    -- fmap takes a function from one type to another and a functor applied
    -- with one type that returns a functor applied with another type.
    -- the canonical map is just an fmap that works only on lists.
    fmap' :: (a -> b) -> f a -> f b

-- trivail to implement for a list, which already has map.
instance Functor' [] where
    fmap' = map

instance Functor' Maybe where
    fmap' f (Just x) = Just (f x)
    fmap' f Nothing = Nothing

-- Either is a type constructor taking two values: Either a b, 
-- but (Either a) just takes one value (the b).
-- The semantics for mapping are similar to those of Maybe.
instance Functor' (Either a) where
    fmap' f (Right x) = Right (f x) -- mapping over right applies the function to the argument.
    fmap' f (Left x) = Left x -- mapping over left just returns the same thing.