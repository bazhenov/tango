output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.vm.id
}

output "public_ip" {
  description = "Public IP address of the VM"
  value       = aws_instance.vm.public_ip
}

output "ssh_command" {
  description = "Ready-to-use SSH command to connect to the VM"
  value       = "ssh -i ${var.public_key_path} ubuntu@${aws_instance.vm.public_ip}"
}
