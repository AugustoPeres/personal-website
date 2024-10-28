+++
title = "Monte Carlo Tree Search in Haskell"
author = ["Augusto"]
description = """A simple implementation of the Monte Carlo Tree Search algorithm in Haskell
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
engineer I no longer had the opportunity to use very often. As such, out of
nostalgia I decided to learn it again and use it to implement a Monte Carlo Tree
Search (MCTS) algorithm to play, perfect information zero sum adversarial games.

In this blog post we go over my implementation of the algorithm as well as other
fun and unique things regarding the Haskell programming language.

The full implementation can be found on my
[github](https://github.com/AugustoPeres/haskell-AI).

# The Adversarial game type class

## What are type classes

Before diving into the algorithm we first need a the game that will be
played. However, because I want my MCTS algorithm to work for any adversarial
perfect information zero sum game I am going to define a `type class` in
haskell.

`Type classes` in haskell are different from classes in python. They do not
represent objects. Instead they simply ensure that certain operations will exist
for data types of that class. A basic example of type classes is the `Show` type
class. This type class ensure that, for any data type of that class a method
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
functions to write the MCTS algorithm and the game loops will exist. The code
for it is presented below:

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

* `step :: g -> a -> a`: This functions receives a game and an action. Applies
  the action to the game and returns the resulting game.
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
number generation) and returns a game and a new state.

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
class. As such, for all games we do not need to bother with them. We need only
to implement the functions that are not yet implemented.

For a better explanation of this topic take a look at [this
chapter](https://learnyouahaskell.com/for-a-few-monads-more) of the [Learn You a
Haskell for Great Good!](https://learnyouahaskell.com/for-a-few-monads-more)
book.

# Connect Four implementation


