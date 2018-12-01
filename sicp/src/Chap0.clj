
(ns Chap0)

; Mostly examples from this page: https://learnxinyminutes.com/docs/clojure/

(str "Hello" " " "World")                                   ;
(println (str "Hello World"))

(println (+ 1 1))
(println (- 2 1))
(println (* 1 2))
(println (/ 2 1))

(println (= 1 1))
(println (= 2 1))

(def x 1)
(println x)

(defn hello-world [] "Hello world")
(println (hello-world))

(defn hello-person [person] (str "Hello, " person))
(println (hello-person "Alex"))

(defn hello
  ([] "Hello world")
  ([name] (str "Hello " name)))
(println (hello))
(println (hello "Alex"))

(def linkedlist `(1 2 3 4 5))
(println linkedlist)

(def vctr [1 2 3 4 5])
(println vctr)

(def infiniteRange range)
(println (take 5 (infiniteRange)))

(def linkedlistplusplus (conj linkedlist 6))
(println linkedlistplusplus)

(def vctrplusplus (conj vctr 6))
(println vctrplusplus)

(defn timestwo [x] (* x 2))
(def linkedlisttimetwo (map timestwo linkedlist))
(println linkedlisttimetwo)

(def linkedlistsum (reduce + linkedlist))
(println linkedlistsum)

(println (class {:a 1 :b 2 :c 3}))
(println (class (hash-map :a 1 :b 2 :c 3)))

(def keymap {:a 1, :b 2, :c 3})
(println keymap)
(println (keymap :a))

(defn fib [n]
  (if (<= n 1) n
               (+ (fib (- n 1))
                  (fib (- n 2)))))
(println (fib 9))

(defn recmax
  ([seq] (recmax seq 0))
  ([seq, acc] (cond (empty? seq) acc
                    :else (recmax (rest seq) (max (first seq) acc)))))
(println (recmax '(1 2 3 4 5)))

(defn mean
  ([seq] (mean seq 0 0))
  ([seq, runSum, runCnt] (cond (empty? seq) (/ (* runSum 1.0) runCnt)
                               :else (mean (rest seq) (+ (first seq) runSum) (inc runCnt)))))
(println (mean '(1 2 3 4 5)))