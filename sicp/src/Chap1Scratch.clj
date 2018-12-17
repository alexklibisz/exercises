(ns Chap1Scratch)

(defn Example []
  (loop [x 10]
    (when (> x 1)
      (println x)
      (recur (- x 2)))))





;(loop [acc 0 k 0]
;  (cond (= k n) acc
;        (odd? k) (recur (+ acc (* 2 (f (+ a (* k h))))) (inc k))
;        ))

;(defn letloop [a]
;  (let [x (+ a 10)]
;    (loop [i 0]
;      (when (< i x)
;        (println i)
;        (recur (inc i))))))

;(letloop 2)