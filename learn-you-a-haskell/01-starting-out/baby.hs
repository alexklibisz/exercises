doubleMe x = x + x
-- doubleUs x y = x * 2 + y * 2
doubleUs x y = doubleMe x + doubleMe y
doubleSmallNumber x = if x > 100
    then x
    else doubleMe x

conanO'Brien = "It's a-me, Conan O'Brien!"

conanChar n = conanO'Brien !! n

listCompExample = [ x | x <- [50..100], x `mod` 7 == 3 ]

fizzBuzzConvert x = 
    if x `mod` 3 == 0 && x `mod` 5 == 0 then "FizzBuzz"
    else if x `mod` 3 == 0 then "Fizz"
    else if x `mod` 5 == 0 then "Buzz" 
    else show x

fizzBuzz = [ fizzBuzzConvert x | x <- [1..100]]