-- hof = higher order function

-- read: max takes an a and returns a funcation that takes an a and returns an a.
max' :: (Ord a) => a -> (a -> a)
max' x y
    | x > y = x
    | otherwise = y

-- Example of "sectioning". 
divBy10 :: (Floating a) => a -> a
divBy10 = (/10)

-- Another example of curried application.
divByN :: (Floating a) => a -> (a -> a)
divByN m n = (/n) m

-- Example of a function that takes a function.
-- The first set of parens is necessary to denote that the argument is a function.
-- example: applyTwice (* 2) 4 = 16
applyTwice :: (a -> a) -> a -> a
applyTwice f x = f (f x)

-- Example invocations:
-- zipWith' (+) [1,2,3] [8,9,10]
-- zipWith' (\a -> \b -> a + b) [1,2,3] [8,9,10]
zipWith' :: (a -> b -> c) -> [a] -> [b] -> [c]
zipWith' _ [] _ = []
zipWith' _ _ [] = []
zipWith' f (x:xtail) (y:ytail) = f x y : zipWith' f xtail ytail

-- flip takes a function of types (a, b) and turns it into a f'n of types (b, a).
-- Example: flip' zip [1..5] "hello"
flip' :: (a -> b -> c) -> (b -> a -> c)
flip' f y x = f x y

-- ah, finally.
map' :: (a -> b) -> [a] -> [b]
map' _ [] = []
map' f (x:xs) = f x : map' f xs

filter' :: (a -> Bool) -> [a] -> [a]
filter' _ [] = []
filter' p (x:xs)
    | p x = x : filter p xs
    | otherwise = filter p xs

partition' :: (a -> Bool) -> [a] -> ([a], [a])
partition' _ [] = ([], [])
partition' f xs = ptn f xs ([], [])
    -- This is the best way I know how to define a "private" internal function.
    where ptn f [] (ts, fs) = (ts, fs)
          ptn f (x:xs) (ts, fs) = 
              if f x then ptn f xs (x:ts, fs)
              else ptn f xs (ts, x:fs)

quicksort :: (Ord a) => [a] -> [a]
quicksort [] = [] -- empty list is sorted by default.
quicksort (pivot:rest) =
    lteqSorted ++ [pivot] ++ gtSorted
    where (lteq, gt) = partition' (<= pivot) rest
          lteqSorted = quicksort lteq
          gtSorted = quicksort gt 

