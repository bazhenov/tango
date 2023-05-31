set term svg enhanced size 1000,400 lw 1.5
set output 'pair-test.svg'
set grid

set datafile separator ',

set multiplot

set ylabel "time (us.)"
set xlabel "observation no"

set title "Execution time"
set size 0.6,1
set origin 0,0
set yrange [-25:25]
#set xrange [0:700]
plot "result.csv" using ($1/1000) title "base" with linespoints pt 1 ps 0.3 lw 0.2 lc 'dark-red', \
     "result.csv" using (-$2/1000) title "-candidate" with linespoints pt 1 ps 0.3 lw 0.2 lc 'dark-green', \
     "result.csv" using ($3/1000) title "(candidate-baseline)" with lines lw 0.5 lc 'navy'

#set log x 10
#set log y 10

set xtics autofreq
set ytics autofreq

set ylabel "time (us.) - candidate"
set xlabel "time (us.) - base"

f(x) = x

unset title
set size 0.4,1
set origin 0.6,0
set xrange [2:7]
set yrange [2:7]
unset key

set object 1 rect from 2.6,2.6 to 3.5,3.5 lw 0.8 fs empty border lc 'navy'
set object 2 rect from 4.3,4.3 to 5.6,5.6 lw 0.8 fs empty border lc 'navy'

plot f(x) notitle with lines linestyle 1 lc "red" dt 4 lw 1, \
     "result.csv" using ($1/1000):($2/1000) title "time to execute" with points pt 1 ps 0.5 lc rgb 'dark-green'
