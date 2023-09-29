provider "aws" {
  region = "us-west-2"
}

provider "tls" {}

locals {
  suffix   = lower(random_string.suffix.result)
  name     = "quantization-${local.suffix}"
  az_index = 1
  azs      = slice(data.aws_availability_zones.available.names, 0, 2)
  vpc_cidr = "10.0.0.0/16"
  tags = {
    Project = "quantization"
  }
}

data "aws_availability_zones" "available" {}

resource "random_string" "suffix" {
  length  = 8
  special = false
}

################################################################################
# VPC
################################################################################

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.0.0"

  name = "${local.name}-vpc"

  cidr            = local.vpc_cidr
  azs             = local.azs
  private_subnets = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 8, k)]
  public_subnets  = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 8, k + 3)]

  enable_nat_gateway = true
  single_nat_gateway = true
}

resource "tls_private_key" "this" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "this" {
  key_name   = "${local.name}-kp"
  public_key = tls_private_key.this.public_key_openssh

  tags = local.tags
}

resource "aws_security_group" "this" {
  name        = "${local.name}-sg"
  description = "cluster ssh"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description = "SSH VPC"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    self        = true
  }

  egress {
    description = "SSH VPC"
    from_port   = "22"
    to_port     = "22"
    protocol    = "tcp"
    self        = true
  }

  tags = local.tags
}

resource "aws_security_group" "efa" {
  name        = "EFA-enabled security group"
  description = "Security group for EFA-enabled instances"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"
    self      = true
  }

  egress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"
    self      = true
  }

  tags = {
    Name = "EFA-enabled security group"
  }
}

resource "aws_security_group" "internet" {
  name        = "internet access"
  description = "access the internet"
  vpc_id      = module.vpc.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

data "aws_ami" "ubuntu_gpu" {
  owners = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base GPU AMI (Ubuntu 20.04) 20230719"]
  }

  most_recent = true
}

locals {
  instances = [
    {
      ami           = data.aws_ami.ubuntu_gpu.id
      instance_type = "g5.xlarge"
    }
  ]
}

resource "aws_instance" "this" {
  count             = length(local.instances)
  ami               = local.instances[count.index].ami
  instance_type     = local.instances[count.index].instance_type
  key_name          = aws_key_pair.this.key_name
  availability_zone = local.azs[local.az_index]
  subnet_id         = module.vpc.private_subnets[local.az_index]
  private_ip        = cidrhost(module.vpc.private_subnets_cidr_blocks[local.az_index], count.index + 50)

  vpc_security_group_ids = [
    aws_security_group.this.id,
    aws_security_group.efa.id,
    aws_security_group.internet.id
  ]

  root_block_device {
    volume_size           = "256"
    volume_type           = "gp3"
    delete_on_termination = true
  }

  tags = {
    Name = "${local.name}-${count.index}"
  }
}

# Bastion

resource "aws_eip" "bastion" {}

resource "aws_eip_association" "bastion" {
  instance_id   = aws_instance.bastion.id
  allocation_id = aws_eip.bastion.id
}

resource "tls_private_key" "bastion" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "bastion" {
  key_name   = "${local.name}-bastion-kp"
  public_key = tls_private_key.bastion.public_key_openssh
}

resource "aws_security_group" "bastion" {
  name        = "bastion"
  description = "bastion security group"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description = "SSH from laptop"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # replace this with your IP/CIDR
  }

  tags = {
    Name = "dev-ssh"
  }
}

data "aws_ami" "latest_amazon_linux" {
  most_recent = true

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*"]
  }

  filter {
    name   = "owner-alias"
    values = ["amazon"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }

  filter {
    name   = "root-device-type"
    values = ["ebs"]
  }

  owners = ["amazon"]
}

resource "aws_instance" "bastion" {
  instance_type = "t3.micro"
  key_name      = aws_key_pair.bastion.key_name
  subnet_id     = module.vpc.public_subnets[local.az_index]
  ami           = data.aws_ami.latest_amazon_linux.id

  vpc_security_group_ids = [
    aws_security_group.bastion.id,
    aws_security_group.this.id
  ]

  tags = {
    Name = "${local.name}-bastion"
  }
}

# Outputs

output "bastion_private_key" {
  description = ""
  value       = tls_private_key.bastion.private_key_pem
  sensitive   = true
}

output "public_ip" {
  description = ""
  value       = aws_eip.bastion.public_ip
}

output "private_key" {
  description = ""
  value       = tls_private_key.this.private_key_pem
  sensitive   = true
}

output "node_ips" {
  description = ""
  value       = aws_instance.this[*].private_ip
}

output "node_ids" {
  description = ""
  value       = aws_instance.this[*].id
}

