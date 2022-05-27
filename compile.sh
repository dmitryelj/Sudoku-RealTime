#!/bin/bash

if [[ "$OSTYPE" =~ ^darwin ]]; then
    echo Building 'solver_lib.so' for OSX...
    gcc -shared -Wl,-install_name,solver_lib.so -o solver_lib.so -fPIC solver_lib.c
    echo Done
fi
if [[ "$OSTYPE" =~ ^linux ]]; then
    echo Building 'solver_lib.so' for Linux...
    gcc -shared -Wl,-soname,solver_lib -o solver_lib.so -fPIC solver_lib.c
    echo Done
fi
