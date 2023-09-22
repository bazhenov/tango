#!/bin/bash

gnuplot -c plot.gnuplot "$1" "plot.svg" && osascript -e 'tell application "Google Chrome" to tell the active tab of its first window to reload'
