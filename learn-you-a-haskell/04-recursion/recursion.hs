
maximum' :: (Ord a) => [a] -> a
maximum' [] = error "maximum of empty list"
maximum' [x] = x -- max of a singleton list.
maximum' (head : tail)
    | head > maxTail = head
    | otherwise = maxTail
    where maxTail = maximum' tail


replicate' :: (Num i, Ord i) => i -> a -> [a]
replicate' n x
    | n <= 0 = []
    | otherwise = x:replicate' (n -1) x

quicksort :: (Ord a) => [a] -> [a]
quicksort [] = [] -- empty list is sorted by default.
quicksort (pivot:rest) =
    lteqSorted ++ [pivot] ++ gtSorted
    where  lteqSorted = quicksort [a | a <- rest, a <= pivot]
           gtSorted = quicksort [a | a <- rest, a > pivot]
    