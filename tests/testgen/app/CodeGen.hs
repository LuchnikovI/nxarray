module CodeGen(compareFinalNXArrays) where

import Control.Monad.State
import ExprTree(genRandomExprTree, genPyCode, rebuildWithRandomRotations, GeneratorState, varsNum)

shebang :: String
shebang = "#!/usr/bin/env python3\n\n"

imports :: String
imports = "from numpy.random import normal\nfrom numpy import isclose\nfrom nxarray import NXArray\n\n"

funcHeader :: String
funcHeader = "def test_autogen():\n"

assertEq :: (Show l, Show r) => l -> r -> String
assertEq lhs rhs =
    let
        lhsVar = ("x" ++ show lhs)
        rhsVar = ("x" ++ show rhs)
    in
        "\tassert set(" ++ lhsVar ++ ".index_ids) == set(" ++ rhsVar ++ ".index_ids)\n" ++
        "\tassert isclose(" ++ lhsVar ++ ".release_normalized_array(*" ++ lhsVar ++ ".index_ids), " ++ rhsVar ++ ".release_normalized_array(*" ++ lhsVar ++ ".index_ids)).all()\n"
        -- ++ "\tassert " ++ lhsVar ++ " == " ++ rhsVar ++ "\n"

separator :: String
separator = "\n# ----------------------------------------------------------------------------------- #\n\n"

printOk :: String
printOk = "\tprint(\"Test final arrays comparison: Ok\")\n\n"

funcCall :: String
funcCall = "if __name__ == \"__main__\":\n\ttest_autogen()\n"

compareFinalNXArrays :: GeneratorState -> String
compareFinalNXArrays = evalState helper where
        helper = do
            arrId1 <- gets varsNum
            tree <- genRandomExprTree
            treeRebuilt <- rebuildWithRandomRotations tree
            arrId2 <- gets ((\x -> x - 1) . varsNum)
            return $
                shebang ++
                imports ++
                funcHeader ++
                genPyCode tree ++
                separator ++
                genPyCode treeRebuilt ++
                assertEq arrId1 arrId2 ++
                printOk ++
                funcCall
        