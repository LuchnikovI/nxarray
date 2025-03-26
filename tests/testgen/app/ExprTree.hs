{-# LANGUAGE FlexibleInstances #-}

module ExprTree(mkGeneratorState, genRandomExprTree, genPyCode, rebuildWithRandomRotations, GeneratorState, Generator, varsNum) where

import System.Random
import Control.Monad.State
import Data.List ((\\))

type NamedIndex = (Int, String)

type NamedIndices = [NamedIndex]

data GeneratorState = GeneratorState {
    rng :: StdGen,
    maxDepth :: Int,
    maxRank :: Int,
    maxLocalDim :: Int,
    maxElementsNum :: Int,
    varsNum :: Int,
    nextId :: Int
} deriving Show

mkGeneratorState :: Int -> Int -> Int -> Int -> Int -> GeneratorState
mkGeneratorState seed md mr mld men = GeneratorState (mkStdGen seed) md mr mld men 0 0

type Generator a = State GeneratorState a

proposeNamedIndex :: Generator NamedIndex
proposeNamedIndex = do
    dimThrshld <- gets maxLocalDim
    newId <- gets nextId
    (localDim, newRng) <- gets (randomR (1, dimThrshld) . rng)
    modify (\st -> st { rng = newRng, nextId = newId + 1 })
    return (localDim, "i" ++ show newId)

_genNamedIndices :: Int -> Int -> Generator NamedIndices
_genNamedIndices 0 _ = return []
_genNamedIndices _ 0 = return []
_genNamedIndices rankThrshld elemNumTrshld = do
    namedIndex <- proposeNamedIndex
    (namedIndex :) <$> _genNamedIndices (rankThrshld - 1) (elemNumTrshld `div` fst namedIndex)

genNamedIndices :: Generator NamedIndices
genNamedIndices = do
    rankThrshld <- gets maxRank
    elemNumThrshld <- gets maxElementsNum
    (randRank, newRng) <- gets (randomR (0, rankThrshld) . rng)
    modify (\st -> st { rng = newRng })
    _genNamedIndices randRank elemNumThrshld


splitSeq :: [a] -> Generator ([a], [a])
splitSeq s = do
    (pos, newRng) <- gets (randomR (0, length s) . rng)
    modify (\st -> st { rng = newRng })
    return (splitAt pos s)

shuffle :: [a] -> Generator [a]
shuffle lst = do
    parts <- splitSeq lst
    case parts of
        ([], []) -> return []
        (x : xs, []) -> (x :) <$> shuffle xs
        (xs, y : ys) -> (y :) <$> shuffle (xs ++ ys)

contract :: (Eq a) => [a] -> [a] -> [a]
contract lhs rhs = (lhs \\ rhs) ++ (rhs \\ lhs)

genMoreNamedIndices :: NamedIndices -> NamedIndices -> Generator NamedIndices
genMoreNamedIndices lhs rhs = do
        let elemNum = max (product (fst <$> lhs)) (product (fst <$> rhs))
        let rank = max (length lhs) (length rhs)
        elemNumThrshld <- gets maxElementsNum
        rankThrshld <- gets maxRank
        _genNamedIndices (rankThrshld - rank) (elemNumThrshld `div` elemNum)


genChildNamedIndices :: NamedIndices -> Generator (NamedIndices, NamedIndices)
genChildNamedIndices namedIndices = do
    (lhs, rhs) <- splitSeq namedIndices
    more <- genMoreNamedIndices lhs rhs
    (,) <$> shuffle (more ++ lhs) <*> shuffle (more ++ rhs)


data ExprTree = ExprTree {
    getExprType :: ExprType,
    getDepth :: Int,
    getVarName :: String,
    getNamedIndices :: NamedIndices
} deriving Show

data ExprType = Mul ExprTree ExprTree | NXArray | Binding String deriving Show

infixl 7 :*:
data TreeStruct a = TreeStruct a :*: TreeStruct a | Root a deriving (Show, Eq)

treeMixer :: TreeStruct ExprTree -> Generator ExprTree
treeMixer (Root tree) = return tree
treeMixer (lhs :*: rhs) = do
    varId <- gets varsNum
    modify (\st -> st { varsNum = varId + 1 })
    built_lhs <- treeMixer lhs
    built_rhs <- treeMixer rhs
    return $ ExprTree
        (Mul built_lhs built_rhs)
        (1 + max (getDepth built_lhs) (getDepth built_rhs))
        ("x" ++ show varId)
        (contract (getNamedIndices built_lhs) (getNamedIndices built_rhs))

rebuildWithRandomRotations :: ExprTree -> Generator ExprTree
rebuildWithRandomRotations tree@ExprTree { getExprType = NXArray, getVarName = varName } = do
    varId <- gets varsNum
    modify (\st -> st { varsNum = varId + 1 })
    return $ tree { getExprType = Binding varName, getVarName = "x" ++ show varId }
rebuildWithRandomRotations tree@ExprTree { getExprType = bind@(Binding _) } = do
    varId <- gets varsNum
    modify (\st -> st { varsNum = varId + 1 })
    return $ tree { getExprType = bind, getVarName = "x" ++ show varId }
rebuildWithRandomRotations ExprTree{ getExprType = (Mul lhs rhs)} = do
    (decisionVar, newRng) <- gets (randomR (0 :: Int, 2) . rng)
    modify (\st -> st { rng = newRng })
    case decisionVar of
        0 -> do
            lhs_rebuilt <- rebuildWithRandomRotations lhs
            rhs_rebuilt <- rebuildWithRandomRotations rhs
            treeMixer (Root lhs_rebuilt :*: Root rhs_rebuilt)
        1 -> do
            case lhs of
                ExprTree{ getExprType = (Mul llhs lrhs)} -> do
                    llhs_rebuilt <- rebuildWithRandomRotations llhs
                    lrhs_rebuilt <- rebuildWithRandomRotations lrhs
                    rhs_rebuilt <- rebuildWithRandomRotations rhs
                    treeMixer (Root llhs_rebuilt :*: (Root lrhs_rebuilt :*: Root rhs_rebuilt))
                _ -> do
                    lhs_rebuilt <- rebuildWithRandomRotations lhs
                    rhs_rebuilt <- rebuildWithRandomRotations rhs
                    treeMixer (Root lhs_rebuilt :*: Root rhs_rebuilt)
        2 -> do
            case rhs of
                ExprTree{ getExprType = (Mul rlhs rrhs)} -> do
                    rlhs_rebuilt <- rebuildWithRandomRotations rlhs
                    rrhs_rebuilt <- rebuildWithRandomRotations rrhs
                    lhs_rebuilt <- rebuildWithRandomRotations lhs
                    treeMixer ((Root lhs_rebuilt :*: Root rlhs_rebuilt) :*: Root rrhs_rebuilt)
                _ -> do
                    lhs_rebuilt <- rebuildWithRandomRotations lhs
                    rhs_rebuilt <- rebuildWithRandomRotations rhs
                    treeMixer (Root lhs_rebuilt :*: Root rhs_rebuilt)
        _ -> undefined

genRandomNXArray :: Generator ExprTree
genRandomNXArray = do
    varId <- gets varsNum
    modify (\st -> st { varsNum = varId + 1 })
    ExprTree NXArray 1 ("x" ++ show varId) <$> genNamedIndices

genRandomExprTree :: Generator ExprTree
genRandomExprTree = do
    top  <- genRandomNXArray
    depthThrshld <- gets maxDepth
    let helper :: ExprTree -> Generator ExprTree
        helper et@(ExprTree NXArray depth _ namedIndices) | depth < depthThrshld = do
            (lhsNamedIndices, rhsNamedIndices) <- genChildNamedIndices namedIndices
            varId <- gets varsNum
            modify (\st -> st { varsNum = varId + 2 })
            newExprType <-
                    Mul <$> helper (ExprTree NXArray (depth + 1) ("x" ++ show varId) lhsNamedIndices)
                        <*> helper (ExprTree NXArray (depth + 1) ("x" ++ show (varId + 1)) rhsNamedIndices)
            return et{ getExprType = newExprType }
                                                          | otherwise = return et
        helper _ = undefined
        in
            helper top

class PyCode c where
    genPyCode :: c -> String

instance PyCode [Int] where
    genPyCode shape = '(' : foldr (\x acc -> show x ++ ',' : acc) ")" shape

instance PyCode [String] where
    genPyCode = foldr (\x acc -> "\"" ++ x ++ "\"," ++ acc) ""

instance PyCode ExprTree where
    genPyCode (ExprTree NXArray _ varName namedIndices) =
        let shape = genPyCode (fst <$> namedIndices)
            names = genPyCode (snd <$> namedIndices)
        in
            "\t" ++ varName ++ " = NXArray(normal(size = " ++ shape ++ "), " ++ names ++ ")\n"
    genPyCode (ExprTree (Mul lhs rhs) _ varName namedIndices) =
        let
            left_code = genPyCode lhs
            right_code = genPyCode rhs
            leftVarName = getVarName lhs
            rightVarName = getVarName rhs
            names = snd <$> namedIndices
            shape = fst <$> namedIndices
            rank = length shape
        in
            left_code ++ right_code ++ "\t" ++ varName ++ " = " ++ leftVarName ++ " * " ++ rightVarName ++ "\n" ++
            "\t" ++ "assert set(" ++ varName ++ ".index_ids) == set([" ++ genPyCode names ++ "])\n" ++
            "\t" ++ "assert " ++ varName ++ ".release_array(" ++ genPyCode names ++ ").shape == " ++ genPyCode shape ++ "\n" ++
            "\t" ++ "assert " ++ varName ++ ".rank == " ++ show rank ++ "\n"
    genPyCode (ExprTree (Binding bindName) _ varName _) = "\t" ++ varName ++ " = " ++ bindName ++ "\n"