- t3.micro

```
./criterion --save-baseline=main --bench -n > /dev/null

while true; do
    ./criterion --bench --baseline=main --noise-threshold=0.005 --measurement-time=1 -n -v >> ./criterion.txt

    ./criterion --save-baseline=paired --noise-threshold=0.005 --measurement-time=1 --bench -n > /dev/null
    ./criterion --bench --baseline=paired --noise-threshold=0.005 --measurement-time=1 -n -v >> ./criterion-paired.txt

    ./tango-1 compare ./tango-2 -t 1 -p --sampler=flat >> ./tango.txt
done
```
