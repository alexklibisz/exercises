
f :: Int -> Int
f x = x * x

g :: Int -> Int
g x = x + 1

h :: Int -> Int
h x = f (g x)

i :: Int -> Int
i x = g (f x)

main = do
    print (h 2) -- apply g, then f to get 9
    print (i 2) -- apply f, then g to get 5