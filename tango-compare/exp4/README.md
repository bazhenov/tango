- t3.micro

```
./criterion --save-baseline=main --confidence-level=0.99 --bench -n > /dev/null

while true; do
    ./criterion --bench --baseline=main --confidence-level=0.99 --measurement-time=1 -n -v >> ./criterion.txt

    ./criterion --save-baseline=in-place --confidence-level=0.99 --measurement-time=1 --bench -n > /dev/null
    ./criterion --bench --baseline=in-place --confidence-level=0.99 --measurement-time=1 -n -v >> ./criterion-in-place.txt

    ./tango-1 compare ./tango-2 -t 1 -p --sampler=flat >> ./tango.txt
done
```
