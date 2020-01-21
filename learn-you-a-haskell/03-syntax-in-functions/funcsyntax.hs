-- An example of pattern matching.
lucky :: (Integral a) => a -> String
lucky 7 = "Lucky number seven!"
lucky x = "Sorry, you're out of luck!"

-- Factorial with pattern matching.
fac :: (Integral a) => a -> a
fac 0 = 1
fac n = n * fac (n - 1)

addVecs :: (Num a) => (a, a) -> (a,a) -> (a,a)
addVecs (x1,y1) (x2,y2) = (x1 + x2, y1 + y2)

-- Use pattern matching to compute product of list recursively.
myProd :: (Num a) => [a] -> a
myProd [] = error "Can't compute product of empty list" 
myProd l = myProd2 (l, 1)
-- myProd [] = 1
-- myProd (n : rest) = n * (myProd rest)

myProd2 :: (Num a) => ([a], a) -> a
myProd2 ([], p) = p
myProd2 (x : xs, acc) = myProd2 (xs, acc * x)

bmiCalc :: (RealFloat a) => a -> a -> a
bmiCalc weight height = weight / height ^ 2

bmiSay :: (RealFloat a) => a -> a -> String
bmiSay weight height
    | bmi <= skinny = "You're underweight, you emo, you!"
    | bmi <= normal = "You're supposedly normal. Pfft, I bet you're ugly!"
    | bmi <= fat = "You're fat! Lose some weight fatty!"
    | otherwise = "You're a whale, congratulations!"
    where bmi = bmiCalc weight height
          (skinny, normal, fat) = (18.5, 25, 30) -- Use a pattern match instead of three lines.

calcBmis :: (RealFloat a) => [(a, a)] -> [a]
calcBmis xs = [bmiCalc w h | (w, h) <- xs]

calcBmis2 :: (RealFloat a) => [(a, a)] -> [a]
calcBmis2 xs = [bmi | (w, h) <- xs, let bmi = w / h ^ 2]

cylinderSurface :: (RealFloat a) => a -> a -> a
cylinderSurface r h = 
    let sideArea = 2 * pi * r * h
        topArea = pi * r^2
    in sideArea + 2 * topArea

-- Pattern matching with case expressions. 
describeList :: [a] -> String
describeList xs = "The list is " ++ case xs of [] -> "empty."
                                               [x] -> "a singleton list."
                                               xs -> "a longer list."
                                               