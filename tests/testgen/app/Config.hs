{-# LANGUAGE OverloadedStrings #-}

module Config where
import Data.Yaml (FromJSON(..), (.:))
import qualified Data.Yaml as Y
import ExprTree(GeneratorState, mkGeneratorState)

data Config = Config {
    getSeed :: Int,
    getMaxDepth :: Int,
    getMaxRank :: Int,
    getMaxLocalDim :: Int,
    getMaxElementsNum :: Int
} deriving (Show, Eq)

config2GeneratorState :: Config -> GeneratorState
config2GeneratorState (Config seed maxDepth maxRank maxLocalDim maxElementsNum) = mkGeneratorState
    seed
    maxDepth
    maxRank
    maxLocalDim
    maxElementsNum


instance FromJSON Config where
    parseJSON (Y.Object v) =
        Config <$>
            v .:   "seed"                <*>
            v .:   "max_depth"           <*>
            v .:   "max_rank"            <*>
            v .:   "max_local_dimension" <*>
            v .:   "max_elements_number"
    parseJSON _ = fail "Expected Object for Config value"