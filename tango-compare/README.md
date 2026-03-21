# Tango Compare

Tools for objectively comparing [Criterion](https://github.com/bheisler/criterion.rs) and Tango benchmark harnesses.

## Methodology

Both harnesses run the same benchmark on the same machine: decoding a UTF-8 string composed of equal numbers of 1-byte, 2-byte, 3-byte, and 4-byte characters (see `gen_utf8.rs`). Criterion and Tango benchmarks alternate in the same loop.

Toolchain is created to do comparison in the AWS cloud, but can be hacked to run anywhere.

## Prerequisites

- `aws` CLI installed and authenticated

## How to run

1. Link an SSH keypair into `terraform/.ssh`

   ```console
   $ mkdir terraform/.ssh
   $ ln -s ~/.ssh/aws-key terraform/.ssh/key
   $ ln -s ~/.ssh/aws-key.pub terraform/.ssh/key.pub
   ```

2. Provision infrastructure

   ```console
   $ terraform init
   $ terraform apply
   ```

3. Run benchmarks on the target machine

   ```console
   $ ssh -i .ssh/key ubuntu@[AWS-IP]
   $ cd tango
   $ screen
   $ ./tango-compare/run.sh
   # Ctrl-A, D to detach from screen
   ```

4. Copy results locally once enough data has been gathered

   ```console
   $ scp -i .ssh/key ubuntu@[AWS-IP]:~/tango/target/benches/*.txt ./
   ```

5. Run the analysis script

   ```console
   $ ./tango-compare/analyze.sh
   ```
