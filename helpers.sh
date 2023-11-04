#!/usr/bin/env bash

function wvfccode()
{
    poetry run code .
}

function wvfctest()
{
    poetry run python -m estm --source_tiles=samples/Flowers.png --output=out.png
}