TERRAFORM = terraform -chdir=deploy

TF_OUT = tfoutput.json
$(TF_OUT): deploy/terraform.tfstate
	$(TERRAFORM) output -json > $(TF_OUT)

ssh-add: $(TF_OUT)
	echo "Adding keys locally..."
	cat $(TF_OUT) | jq -r .private_key.value | ssh-add -
	cat $(TF_OUT) | jq -r .bastion_private_key.value | ssh-add -

stop-instances:
	aws ec2 stop-instances --instance-ids $(shell cat $(TF_OUT) | jq -r .node_ids.value[0])

start-instances:
	aws ec2 start-instances --instance-ids $(shell cat $(TF_OUT) | jq -r .node_ids.value[0])

