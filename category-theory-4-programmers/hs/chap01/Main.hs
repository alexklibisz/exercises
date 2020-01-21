compose :: (b -> c) -> (a -> b) -> (a -> c)
compose f g = \x -> f (g x)

f = \x -> x * x
g = \x -> x + 1

i = compose f g
h = compose g f

-- TODO: Why doesn't this work? How would you define an infix operator?
-- (andThen) :: (b -> c) -> (a -> b) -> (a -> c)
-- (andThen) f g = \x -> f (g x)
-- j = f andThen g
-- k = g andThen f

main = do
    print (i 2)
    print (h 2)
    -- print (j 2)
    -- print (k 2)