
data Tree a = EmptyTree | Node a (Tree a) (Tree a) deriving (Show, Eq)

singleton :: a -> Tree a
singleton x = Node x EmptyTree EmptyTree

insert :: (Ord a) => a -> Tree a -> Tree a
insert x EmptyTree = singleton x
insert x (Node a l r)
    | x == a = Node x l r
    | x < a = Node a (insert x l) r
    | x > a = Node a l (insert x r)

contains :: (Ord a) => a -> Tree a -> Bool
contains x EmptyTree = False
contains x (Node a l r)
    | x == a = True
    | x < a = contains x l
    | x > a = contains x r

fromList :: (Ord a) => [a] -> Tree a
fromList [] = EmptyTree
fromList xs = foldr insert EmptyTree xs

-- *Main> fmap (*2) (fromList [1,2,3])
-- Node 6 (Node 4 (Node 2 EmptyTree EmptyTree) EmptyTree) EmptyTree
instance Functor Tree where
    fmap f EmptyTree = EmptyTree
    fmap f (Node x l r) = Node (f x) (fmap f l) (fmap f r)