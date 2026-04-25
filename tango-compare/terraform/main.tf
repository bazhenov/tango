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
  instance_type          = var.instance_type
  key_name               = aws_key_pair.deployer.key_name
  vpc_security_group_ids = [aws_security_group.vm_sg.id]

  user_data = templatefile("${path.module}/init.sh", {
    aws_access_key_id     = var.aws_access_key_id
    aws_secret_access_key = var.aws_secret_access_key
    aws_region            = var.aws_region
  })

  root_block_device {
    volume_size = 8 # GiB, stays within free tier (30 GiB/month allowance)
    volume_type = "gp2"
  }

  tags = {
    Name    = "${var.project_name}-vm"
    Project = var.project_name
  }
}
