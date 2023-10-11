#!/bin/bash

# Run reproduce_paper.py and check if it was successful
if python3 code/reproduce_paper.py; then
    echo "reproduce_paper.py ran successfully. Now running myst build..."

    # Run myst build command and check if it was successful
    if myst build  --pdf; then
        echo "myst build completed successfully."
    else
        echo "myst build failed."
    fi
else
    echo "reproduce_paper.py failed to run."
fi
