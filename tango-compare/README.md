# Tango compare SDK

Criterions is well established benchmark harness in the Rust ecosustem. Here is the set of tools to objectivley compare Criterion with Tango.

## Methodology

Comparison is done on the same machine on the same benchmark.

## How to run

1. link SSH-keypair into `terraform/.ssh`

```console
$ mkdir terraform/.ssh
$ ln -s ~/.ssh/aws-key terraform/.ssh/key
$ ln -s ~/.ssh/aws-key.pub terraform/.ssh/key.pub
```

```
$ terraform init
```
