variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "eu-central-1"
}

variable "project_name" {
  description = "Name prefix used for all created resources"
  type        = string
  default     = "tango-compare-vm"
}

variable "ami_id" {
  description = <<-EOT
    AMI ID for Ubuntu 24.04 LTS.
    Free-tier eligible. Region-specific — check https://cloud-images.ubuntu.com/locator/ec2/
    Default is us-east-1.
  EOT
  type        = string
  default     = "ami-005f97cc4a61dd3b4" # Ubuntu Server 24.04 LTS (HVM), SSD Volume Type
}

variable "public_key_path" {
  description = "Path to your local SSH public key, e.g. ~/.ssh/id_ed25519.pub"
  type        = string
  default     = "~/.ssh/aws.pub"
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH into the instance. Restrict to your IP for security, e.g. '1.2.3.4/32'"
  type        = string
  default     = "0.0.0.0/0"
}
