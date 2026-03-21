provider "aws" {
  region = var.aws_region
}

# --- Key Pair ---
resource "aws_key_pair" "deployer" {
  key_name   = "${var.project_name}-key"
  public_key = file(var.public_key_path)
}

# --- Security Group ---
resource "aws_security_group" "vm_sg" {
  name        = "${var.project_name}-sg"
  description = "Allow SSH inbound, all outbound"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.project_name}-sg"
    Project = var.project_name
  }
}

# --- EC2 Instance (free tier: t2.micro) ---
resource "aws_instance" "vm" {
  ami                    = var.ami_id
  instance_type          = "t2.micro"
  key_name               = aws_key_pair.deployer.key_name
  vpc_security_group_ids = [aws_security_group.vm_sg.id]

  # Installs rustup non-interactively on first boot
  user_data = <<-EOF
    #!/bin/bash
    set -euo pipefail

    # Update packages
    apt-get update -y
    apt-get install -y curl

    # Install rustup for the default non-root user (ubuntu)
    sudo -u ubuntu bash -c '
      curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
      echo "source \$HOME/.cargo/env" >> /home/ubuntu/.bashrc
      echo "source \$HOME/.cargo/env" >> /home/ubuntu/.profile
    '
  EOF

  root_block_device {
    volume_size = 8 # GiB, stays within free tier (30 GiB/month allowance)
    volume_type = "gp2"
  }

  tags = {
    Name    = "${var.project_name}-vm"
    Project = var.project_name
  }
}
