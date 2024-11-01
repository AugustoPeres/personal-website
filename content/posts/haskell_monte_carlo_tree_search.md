+++
title = "Monte Carlo Tree Search in Haskell"
author = ["Augusto"]
description = """
A simple implementation of the Monte Carlo Tree Search algorithm in Haskell

```haskell
mctsIteration :: (AdversarialGame g a p, Eq a) => MCTSAgent g a p
                                               -> State StdGen (MCTSAgent g a p)
mctsIteration agent = (selection agent)
                    >>= (return . expansion)
                    >>= simulation
                    >>= (return . backpropagation)
```

  """
draft = false
tags = ["haskell", "AI"]
showFullcontent = false
readingTime = true
hideComments = false
mathJaxSupport = true
+++

# Introduction

During my days as a mathematics graduate student I played around with
[haskell](https://www.haskell.org/): a pure, functional programming language. I
have always found it fascinating and, in fact used it for my master thesis. A
gamble that luckily worked out fine.

However, after finishing my degree and finding a job as machine learning
engineer I no longer had the opportunity to use it very often. As such, out of
nostalgia I decided to learn it again and use it to implement a Monte Carlo Tree
Search (MCTS) algorithm to play, perfect information, zero sum, adversarial
games.

In this blog post we go over my implementation of the algorithm as well as other
fun and unique things regarding the Haskell programming language.

The full implementation can be found on my
[github](https://github.com/AugustoPeres/haskell-AI).

# The Adversarial game type class

## What are type classes

Before diving into the algorithm we first need the game that will be
played. However, because I want my MCTS algorithm to work for any adversarial
perfect information zero sum game I am going to define a `type class` in
haskell.

`Type classes` in haskell are different from classes in python. They do not
represent objects. Instead they simply ensure that certain operations will exist
for data types of that class. A basic example of type classes is the `Show` type
class. This type class ensure that, for any data type of that class a function
will exist that converts it to a string:

```haskell
class Show a where
  show :: a -> String
```

The code above states that for any data type of the `Show` class there will be a
function called `show` that converts it to a string. That means that I can write
the function:

```haskell
joinStrings :: (Show a, Show b) => a -> b -> String
joinstrings arg1 arg2 = show arg1 ++ " " ++ show arg2
```

This function receives any two arguments that derive `Show` and merges their
string representations. The only restriction that we had to express in the
function type signature is that `a` and `b` must both elements that derive
show. For example, because `Int` and `Bool` are both data types deriving `Show`
we can do something like:

```haskell
joinStrings 1 True -- This yeilds "1 True"
```

## Our type class

Our type class will ensure that, for every game deriving it, the necessary
functions to write the MCTS algorithm and the game loops will exist. The class
is defined below:

```haskell
class (Eq p) =>
      AdversarialGame g a p
  | g -> a p
  where
  step :: g -> a -> g
  availableActions :: g -> [a]
  currentPlayer :: g -> p
  isFinal :: g -> Bool
  winner :: g -> Maybe p
  getRandomAction :: g -> State StdGen (Maybe a)
  getRandomAction game =
    if null actions
      then return Nothing
      else get >>= \gen ->
             let (newGen, action) = choice actions gen
              in put newGen >> return (Just action)
    where
      actions = availableActions game
  playRandomAction :: g -> State StdGen g
  playRandomAction game = do
    getRandomAction game
      >>= (\a ->
             case a of
               Nothing     -> return game
               Just action -> return $ step game action)
  playRandomGame :: g -> State StdGen g
  playRandomGame game =
    playRandomAction game >>=
    (\g ->
       case isFinal g of
         True -> return g
         _    -> playRandomGame g)
```

Lets break down this type class:

* `step :: g -> a -> a`: This functions receives a game and an action. Returns
  the result of applying that action to the game.
* `availableActions :: g -> [a]`: This function receives a game and returns a
  list of the legal actions for that game state.
* `currentPlayer :: g -> p`: This function return the current player for a given
  game state
* `isFinal :: g -> Bool`: Receives a game and returns `True` if and only if that
  game state is final
* `winner :: g -> Maybe p`: This function receives a game a returns `Nothing` if
  there is no winner or `Just p` if there is a winner for that game state. If
  this is the first time looking at the `Maybe` monad I recommend that you take
  a look at [this](https://wiki.haskell.org/Maybe).
  
Now the type signatures get more complicated and we see something like
`getRandomAction :: g -> State StdGen (Maybe a)`. Why?

Well, haskell is a pure language. That means that functions must always, for the
same input, return the same output. As such, completely random functions are
rarely used, unless we want to be stuck to the `IO` (input and output) monad
that marks computations as non-pure: Eww!!

A work around is to have something like `playRandomAction :: g -> StdGen -> g`
where `StdGen` is a seed for **pure** random number generation. However, because
we want to chain several random actions, we must track the random seed. That
would mean that the type signature would be `playRandomAction :: g -> StdGen ->
(StdGen, g)`, that is, `playRandomAction` receives a game and a random seed,
steps both the game and the seed and then returns the new game and the new seed
as a tuple.

This can quickly become hard to manage. Enter the `State` monad!! Here, `State
StdGen g` is basically a wrapper for `StdGen -> (g, StdGen)`. That is, the state
monad is an abstraction for a function that receives a state (a seed for random
number generation) and returns a game and a new state (a new random seed).

Therefore, using the monad properties we can do very powerful things like:

```haskell
s = playRandomAction initialGame >>= playRandomAction >>= playRandomAction >>= playRandomAction
```

To play 4 random actions. And now we can do something like:

```haskell
-- runState :: State s a -> s -> (a, s)
fst $ runState s (mkStdGen 1)
fst $ runState s (mkStdGen 2)
```

To obtain the result of applying those four actions to the initial game state
using different starting seeds.

Monads are notorious for being incredibly confusing. Therefore, if you did not
understand this, do not linger here. Additionally, notice how those functions
are already implemented using just the previously defined functions of the
class. As such, for all games we do not need to bother with them.

For a better explanation of this topic take a look at [this
chapter](https://learnyouahaskell.com/for-a-few-monads-more) of the [Learn You a
Haskell for Great Good!](https://learnyouahaskell.com/)
book.

# Connect Four implementation

After defining `Adversarial Game` class we can start to implement games deriving
from it with assurances that later on the algorithms we design will work. In
this particular case I made a very simple connect four implementation is
Haskell:

```haskell
data Player = X | O deriving (Eq, Show)

type Action = Int
type Board = Int -> Int -> Maybe Player

data ConnectFour = ConnectFour
  { board       :: Board
  , player      :: Player
  , piecesInCol :: Int -> Int
  }
```

Lets break this down. First we create a new algebraic data type
`Player`. Basically this states that a `Player` is either player `X` or player
`O`. This is a common way to define data types in Haskell. In fact, the boolean
data type is defined as `data Bool = True | False`.

Next we create a new type synonym where an action is simply an
integer. Similarly, we create a new type synonym `Board = Int -> Int -> Maybe
Player`, this states that a board is a function receiving two integer and
returning `Maybe Player`. How do we use this?  Well, if a square in connect four
is unoccupied then the board function returns `Nothing` otherwise it returns
`Just X` or `Just O` depending on the player that occupies that position on the
board.

Finally, we define out connect `ConnectFour` game. This is a simple data type
with records:

* `board`: Tracks the state of the board
* `player`: Tracks the current player
* `piecesInCol :: Int -> Int`. This is a simple helper function to help us track
  how many pieces are in a given column of the board.

If you are unfamiliar with Haskell, this is will become more clear when we see
how to define the initial game state and the other functions. The initial game
can be trivially defined as:

```haskell
initialGame :: ConnectFour
initialGame = ConnectFour (\_ _ -> Nothing) X (\_ -> 0)
```

As we can see, the initial game is simply a `Board` that always return `Nothing`
because everything is empty. The starting player is `X` and `piecesInCol` is a
function that always return `0`.

We are now ready to make `ConnectFour` an element of the `AdversarialGame` type
class. Here we will show only the `step` and `availableActions` functions, for
the full code please refer to my
[github](https://github.com/AugustoPeres/haskell-AI):

```haskell
instance AdversarialGame ConnectFour Action Player where
  step g a =
    if not (a `elem` availableActions g)
    then g
    else g { board = newBoard
           , player = nextPlayer cPlayer
           , piecesInCol = newPiecesInCol}
    where newBoard c r = if c == a && r == row
                         then Just cPlayer
                         else (board g) c r
          newPiecesInCol c = if c == a
                             then (piecesInCol g c) + 1
                             else (piecesInCol g c)
          row = (piecesInCol g) a
          cPlayer = player g

  availableActions g = filter (\x -> (piecesInCol g) x <= 5) [0..6]
```

The `step` function does the following: If the given action is not available,
then we return the game unchanged. Otherwise we return a new instance of
`ConnectFour` where we changed the `board` function to now return, at the
position where the piece is placed, the current player and we changed the
`piecesInCol` to return, for the column where the piece was placed, the previous
number of pieces in that column plus 1.

Because `ConnectFour` is now an element of the `AdversarialGame` type class we
can do something like:

```haskell
ghci> s = playRandomAction initialGame >>= playRandomAction >>= playRandomAction
ghci> fst $ runState s (mkStdGen 1)
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ X O _ X _ 
current player O
```

Great, we are now ready to start implementing the Monte Carlo Tree Search
algorithm AI

# Monte Carlo Tree Search algorithm

We will not dive into an explanation of the MCTS algorithm here. This has been
extensively covered by several different resources online. As such, we will
focus more in its implementation in Haskell.

## The tree class

Here we will focus on implementing the tree structure on which we can later
build the MCTS algorithm. I always consider this to be a great example of how
easy is to define mathematical abstract structures using Haskell.

First, lets think about what type of tree we want to build. Well, we want the
nodes to store the current game states, and we want the edges to store the
actions that lead to that game state. In Haskell this can be achieved in the
one-liner:

```haskell
data Tree crumb value = Leaf value | Node value [(crumb, Tree crumb value)] deriving (Eq, Show, Ord)
```

The previous line states that a `Tree` is either a `Leaf` storing a given value
or it is a `Node` storing a value and a list of tuples `(crumb, Tree)` that are
its list of children and the values of the edges that link to those
children. From this definition we can already do lots of powerful things. For
one, we can already create trees:

```haskell
tree = Node 2 [("a", Leaf 1), ("b", Node 1 [("c", Leaf 0)])]
-- This is the tree:
--          Node 2
--          /     \
--       "a"       "b"
--       /           \
--    Leaf 1        Node 1
--                      |
--                    "c"
--                      |
--                   Leaf 0
```

And, using this powerful recursive definition we can trivially implement
functions that, for example apply a given function to all values of the node:

```haskell
applyFunction :: Tree crumb value -> (value -> b) -> Tree crumb b
applyFunction (Leaf v) f   = Leaf (f v)
applyFunction (Node v c) f =
  Node (f v) (map (\(crum, tree) -> (crum, applyFunction tree f)) c)
```

Basically the function goes down the tree applying f to the values in the
nodes. For example:

```haskell
applyFunction tree (+5)
-- yileds:
--          Node 7
--          /      \
--       "a"        "b"
--       /            \
--    Leaf 6        Node 6
--                     |
--                   "c"
--                     |
--                  Leaf 5
```

## The zipper class

But wait!!! In MCTS algorithms there is a backup step where, after playing
random games from a `Leaf` node we must backup over the tree and update the win
ratios of all its parents. But, how can we do this recursively if, following the
above definition we can only go down the tree?

The answer is to use zippers! For a much better introduction to zippers please
refer to [this chapter](https://learnyouahaskell.com/zippers) in the [Learn You
a Haskell for great good](https://learnyouahaskell.com/chapters) book. But
essentially, zippers are a list of bread-crumbs that we leave behind when going
down the tree that later allow us to go back up the tree again. Or, in Haskell:

```haskell
data Crumb crumb value = Crumb value crumb [(crumb, Tree crumb value)] deriving (Eq, Show, Ord)
type Zipper crumb value = (Tree crumb value, [Crumb crumb value])
```

Basically a `Crumb` stores the value of the parent, the value of the edge
from that parent to the children when we go down the tree, and the remaining
children of the parent node. For example, for the previous defined tree, if we
go down using edge `"a"` we get the zipper:

```haskell
(Leaf 1, Crumb 2 "a" [Node 1 [("c", Leaf 1)]])
```

Using this we can easily go up the tree maintaining its original structure
again. In fact, lets see an example for this short tree library:

```haskell
goUpWith :: Zipper crumb value -> (value -> value -> value) -> Zipper crumb value
goUpWith z@(_, []) _ = z
goUpWith (tree, (Crumb value crum children):xs) f =
  (Node resultingValue ((crum, tree):children), xs) 
  where resultingValue = f (getRootValue tree ) value 
```

This function goes up the zipper while applying a function with signature `value
-> value -> value` to the nodes in the tree. If the crumbs are empty we return
the zipper. Otherwise we go up one step while applying the function. For
example:

```haskell
zipper = (Leaf 1, Crumb 2 "a" [Node 1 [("c", Leaf 1)]])
goUpWith zipper (-)
-- yields:
-- (Node 1 [("a", Leaf 1), ("b", Node 1 [("c", Leaf 0)])], [])
```

This function will allows to easily implement the backup step in the MCTS algorithm.

We will not show here the entire implementation of these function. Instead we
just give the type signatures:

```haskell
makeZipper :: Tree crumb value -> Zipper crumb value
makeTree :: value -> Tree crumb value
followDirection :: (Eq crumb) => Zipper crumb value -> crumb -> Zipper crumb value
goUpWith :: Zipper crumb value -> (value -> value -> value) -> Zipper crumb value
goToTopWith :: Zipper crumb value -> (value -> value -> value) -> Zipper crumb value
addChildren :: Zipper crumb value -> [(crumb, value)] -> Zipper crumb value
followToBottomWith :: (Eq crumb, Ord b) => Zipper crumb value -> (value -> value -> b) -> (value -> Bool) -> State StdGen (Zipper crumb value)
getRootValue :: Tree crumb value -> value
getChildren :: Tree crumb value -> [(crumb, Tree crumb value)]
updateRootValue :: Tree crumb value -> value -> Tree crumb value
```

## The MCTS algorithm

Our type synonyms and algebraic data types:

```haskell
type MCTSNode g = (g, Int, Int) -- (current game state, wins, n simulations)

data MCTSAgent g a p =
  MCTSAgent { zipper         :: Zipper a (MCTSNode g)
            , player         :: p
            , numSimulations :: Int } deriving (Show)

```

Essentially, a tree node will consist of a game state, the number of wins from
that node and the number of simulations from that node. A `MCTSAgent` will store
the game tree (actually the zipper), the player for which it is playing and the
number of simulations to run when reaching a child node.

The first step in each MCTS iteration is to go down the tree selecting the best
child according to the upper confidence bound (UCB). Using our zipper library we
can implement this using:

```haskell
selection :: (AdversarialGame g a p, Eq a) => MCTSAgent g a p -> State StdGen (MCTSAgent g a p)
selection agent =
  followToBottomWith (zipperAgent) f stoppage >>= (\z -> return $ agent { zipper = z })
  where f parent@(gameState, pWins, pVisits) child@(gameState', cWins, cVisits) =
          if currentPlayer gameState == player agent
          then ucb parent child
          else ucb (gameState, pVisits - pWins, pVisits) (gameState', cVisits - cWins, cWins)
        zipperAgent@(tree, _) = zipper agent
        stoppage (gameState, _, _) = length (availableActions gameState) /= length (getChildren tree)
		

ucb :: MCTSNode g -> MCTSNode g -> Float
ucb (_, _, p_visits) (_, c_wins, c_visits)
  | c_visits == 0 = 1 / 0
  | otherwise =
      let c = sqrt 2
          exploration = c * sqrt ((log $ fromIntegral p_visits) / fromIntegral c_visits)
          exploitation = fromIntegral c_wins / fromIntegral c_visits
      in exploration + exploitation
```

Essentially this functions uses the zipper functions defined in the previous
section and the UCB to go down until we find a node that can still be
expanded. Additionally notice how, in the function type signature, we do not
specify that this function is implemented for the `ConnectFour` data type,
instead we only specified that `g` needs to be an element of `AdversarialGame`
type class.

The next step in a MCTS iteration is to, from the previous reached node, expand it:

```haskell
expansion :: (AdversarialGame g a p, Eq a) => MCTSAgent g a p -> MCTSAgent g a p
expansion agent =
  let z@(tree, _) = zipper agent
      (gameState, _, _) = getRootValue (fst z)
  in case availableActions gameState \\ map fst (getChildren tree) of
       []   -> agent
       a:_ -> agent {zipper = followDirection (addChildren z [(a, (step gameState a, 0, 0))]) a}
```

This function checks if there are actions in that node that can be taken. If
that is the case it adds a child node to it and goes down the tree to that node.

Next we simulate random games from that node:

```haskell
simulation :: (AdversarialGame g a p, Eq a) => MCTSAgent g a p -> State StdGen (MCTSAgent g a p)
simulation agent =
  let z@(tree, crumbs) = zipper agent
      (gameState, _, _) = getRootValue (fst z)
      playerAgent = player agent
      simulations = numSimulations agent
      randomGames = replicateM simulations (playRandomGame gameState)
      wonGames = fmap (\games -> sum $ map f games) randomGames
      f = \game -> case winner game of
                     Just w | w == playerAgent -> 1
                     Just _                    -> -1
                     _                         -> 0
  in if winner gameState == (Just $ player agent)
     then return $ agent { zipper = (updateRootValue tree (gameState, 1, 1), crumbs)}
     else wonGames >>= (\wg -> return $ agent { zipper = (updateRootValue tree (gameState, wg, simulations), crumbs) })
```

Here we simulate random games from a `Leaf` node. The games won are scored with
`1`, ties are scored with `0` and losses are scored with `-1`. We update the win
ratio for that leaf and return the agent with its zipper updated.

Finally, we need to go up the tree to propagate this win ratio:

```haskell
backpropagation :: MCTSAgent g a p -> MCTSAgent g a p
backpropagation agent =
  let z@(tree, _) = zipper agent
      (_, wins, sims) = getRootValue tree
      f = \(_, _, _) (gameState, a, b) -> (gameState, a + wins, b + sims)
      newZipper = goToTopWith z f
  in agent { zipper = newZipper }
```

Here we simply use the `goTotopwith` presented before to go all the way up the
tree while updating the win rates of all nodes.

We are now fully prepared to define a MCTS iteration:

```haskell
mctsIteration :: (AdversarialGame g a p, Eq a) => MCTSAgent g a p -> State StdGen (MCTSAgent g a p)
mctsIteration agent = (selection agent) >>= (return . expansion) >>= simulation >>= (return . backpropagation)
```

And from this we can define a function to take an action from any given game
state:

```haskell
takeAction :: (AdversarialGame g a p, Eq p, Eq a) => g -> Int -> State StdGen a
takeAction gameState numIterations =
  get >>= (\gen ->
          let agent = MCTSAgent { zipper = makeZipper (makeTree (gameState, 0, 0)),
                                  player = currentPlayer gameState,
                                  numSimulations = 100 }
              iteratedAgent = foldM (\ag _ -> mctsIteration ag) agent [1..numIterations]
              (agent', newGen) = runState iteratedAgent gen
              (tree, _) = zipper agent'
              f = (\(_, w, n) -> fromIntegral w / fromIntegral n) . getRootValue . snd
              maxChildren = maxValuesBy (getChildren tree) f
              (newGen', action) = choice (map fst maxChildren) newGen
          in put newGen' >> return action)
```

This is it. This is a MCTS implementation in Haskell from scratch. If you want
to play against the AI you can clone my repository. There you have instructions
to run the code. Also, if you want this for your own game, recall that you must
only implement it as a member of the `AdversarialGame` type class and you are
good to go.

 Here we have a sample game play (I am player X).

```bash
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
current player X

Available actions: 
[0,1,2,3,4,5,6]
Enter your action (as an integer): 
4
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ X _ _ 
current player O

AI thinking...
AI chose action: 3
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ O X _ _ 
current player X

Available actions: 
[0,1,2,3,4,5,6]
Enter your action (as an integer): 
3
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ X _ _ _ 
_ _ _ O X _ _ 
current player O

AI thinking...
AI chose action: 0
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ X _ _ _ 
O _ _ O X _ _ 
current player X

Available actions: 
[0,1,2,3,4,5,6]
Enter your action (as an integer): 
4
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ X X _ _ 
O _ _ O X _ _ 
current player O

AI thinking...
AI chose action: 1
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ X X _ _ 
O O _ O X _ _ 
current player X

Available actions: 
[0,1,2,3,4,5,6]
Enter your action (as an integer): 
2
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ X X _ _ 
O O X O X _ _ 
current player O

AI thinking...
AI chose action: 2
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ O X X _ _ 
O O X O X _ _ 
current player X

Available actions: 
[0,1,2,3,4,5,6]
Enter your action (as an integer): 
4
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ X _ _ 
_ _ O X X _ _ 
O O X O X _ _ 
current player O

AI thinking...
AI chose action: 4
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ O _ _ 
_ _ _ _ X _ _ 
_ _ O X X _ _ 
O O X O X _ _ 
current player X

Available actions: 
[0,1,2,3,4,5,6]
Enter your action (as an integer): 
2
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ O _ _ 
_ _ X _ X _ _ 
_ _ O X X _ _ 
O O X O X _ _ 
current player O

AI thinking...
AI chose action: 3
Game Over! The winner is: O
_ _ _ _ _ _ _ 
_ _ _ _ _ _ _ 
_ _ _ _ O _ _ 
_ _ X O X _ _ 
_ _ O X X _ _ 
O O X O X _ _ 
```

There you have it! Not a brilliant game on my part but the AI can both prevent
me from winning and win itself.
