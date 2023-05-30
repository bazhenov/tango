#!/bin/bash

gnuplot $1 && osascript -e 'tell application "Google Chrome" to tell the active tab of its first window to reload'
