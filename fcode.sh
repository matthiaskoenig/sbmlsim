#!/bin/bash
isort src/sbmlsim
black src/sbmlsim

isort tests
black tests
