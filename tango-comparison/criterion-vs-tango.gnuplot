set term svg enhanced size 2000,500 lw 1.5 background rgb 'white'
set output "criterion-vs-tango.svg"
set grid

set datafile separator ','

set xtics autofreq
set ytics autofreq

set ylabel "%"
set xlabel "observation no"

set title "Difference"

#set yrange [-2:15]

#set ytics 1

f(x) = 0

plot f(x) notitle with lines linestyle 1 lc "red" dt 4 lw 1, \
     "data.csv" using ($1) title "Criterion" with linespoints pt 1 ps 0.3 lw 1 lc 'dark-red', \
     "data.csv" using ($2) title "Criterion in-place" with linespoints pt 1 ps 0.3 lw 1 lc 'red', \
     "data.csv" using ($3) title "Tango" with linespoints pt 1 ps 0.3 lw 1 lc 'dark-green'
