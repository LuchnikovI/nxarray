module Cli(CliOptions(..), cliParser) where

import Options.Applicative

data CliOptions = CliOptions {
    getConfigPath :: String,
    getOutputPath :: String
} deriving Show

cliParser :: Parser CliOptions
cliParser = CliOptions
    <$> strOption
        ( long "config-path"
        <> short 'c'
        <> help "Path to the configuration file"
        <> showDefault
        <> value "./test_autogen.yaml"
        <> metavar "PATH" )
    <*> strOption
        ( long "output-path"
        <> short 'o'
        <> help "Path to the output generated test file"
        <> showDefault
        <> value "./test_autogen.py"
        <> metavar "PATH" )