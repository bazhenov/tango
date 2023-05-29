set term svg enhanced size 1000,1000 lw 1.5
set output 'plot.svg'
set grid
set key left top

set datafile separator ',

set multiplot

set log y 10;

set title "Stderr"
set size 0.5,0.25
set origin 0.5,0.25
plot "result.csv" using 4 title "base" with lines lw 1, \
     "result.csv" using 5 title "candidate" with lines lw 1, \
     "result.csv" using 6 title "diff" with lines lw 1

set ylabel "time (ns.)"
set xlabel "observation no"

set title "Base algorithm time"
set size 0.5,0.25
set origin 0.5,0.5
plot "result.csv" using 1 title "time to execute" with linespoints pt 1 ps 0.3 lw 0.2

set title "Candidate algorithm time"
set size 0.5,0.25
set origin 0.5,0.75
plot "result.csv" using 2 title "time to execute" with linespoints pt 1 ps 0.3 lw 0.2

unset log x
unset log y

set title "Diff algorithm time"
set size 0.5,0.25
set origin 0.0,0.25
plot "result.csv" using 3 title "time to execute" with linespoints pt 1 ps 0.3 lw 0.2

set log x 10
set log y 10

set xtics 2
set ytics 2

set ylabel "time (ns.) - candidate"
set xlabel "time (ns.) - base"

f(x) = x

unset title
set size 0.5,0.5
set origin 0.0,0.5
plot "result.csv" using 1:2 title "time to execute" with points pt 1 ps 0.5 lc rgb 'dark-green', \
     f(x) notitle with lines linestyle 1 lc "red" dt 4 lw 0.7
