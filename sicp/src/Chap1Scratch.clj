(ns Chap1Scratch)

(defn square [x] (* x x))
(println (map square (range 1 10)))

(defn sum-of-squares [x y]
  (+ (square x) (square y)))
(println (sum-of-squares 3 4))

(defn abs [x]
  (cond (>= x 0) x
        (< x 0) (- x)))
(println (abs 10))
(println (abs -99))

(def tup '(1 2))
(println (nth tup 0))
(println (nth tup 1))

(def x 6)
(println (and (> x 5) (< x 10)))