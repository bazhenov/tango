set term svg enhanced size 1200,400 lw 1.5
set output ARG2
set grid

set datafile separator ',

set multiplot

set ylabel "time (us.)"
set xlabel "observation no"

set title "Execution time"
set size 0.6,1
set origin 0,0
plot ARG1 using ($1/$3 / 1000) title "base" with linespoints pt 1 ps 0.3 lw 0.8 lc 'dark-red', \
     ARG1 using (-$2/$3 / 1000) title "-candidate" with linespoints pt 1 ps 0.3 lw 0.8 lc 'dark-green', \
     ARG1 using (($2 - $1) / $3 / 1000) title "(candidate-baseline)" with lines lw 0.8 lc 'navy'

set xtics autofreq
set ytics autofreq

set ylabel "time (us.) - candidate"
set xlabel "time (us.) - base"

f(x) = x

unset title
set size 0.4,1
set origin 0.6,0
unset key

plot f(x) notitle with lines linestyle 1 lc "red" dt 4 lw 1, \
     ARG1 using ($1 / $3 / 1000):($2 / $3 / 1000) title "time to execute" with points pt 1 ps 0.5 lc rgb 'dark-red'
