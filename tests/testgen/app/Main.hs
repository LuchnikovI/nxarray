{-# LANGUAGE OverloadedStrings #-}

module Main where

import CodeGen(compareFinalNXArrays)
import System.Directory (createDirectoryIfMissing)
import System.FilePath.Posix (takeDirectory)
import Data.ByteString(readFile)
import Options.Applicative (execParser, info, helper, fullDesc, progDesc, (<**>))
import Cli (cliParser, CliOptions (CliOptions))
import Data.Yaml (decodeThrow)
import Config (Config, config2GeneratorState)

main :: IO ()
main = executor =<< execParser opts
    where
        opts = info (cliParser <**> helper)
            (fullDesc
            <> progDesc "Generates tests for NXArray library")

executor :: CliOptions -> IO ()
executor (CliOptions configPath outputPath) = do
    configYaml <- Data.ByteString.readFile configPath
    config <- decodeThrow configYaml :: IO Config
    let code = compareFinalNXArrays $ config2GeneratorState config
    createDirectoryIfMissing True $ takeDirectory outputPath
    writeFile outputPath code