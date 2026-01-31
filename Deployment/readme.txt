#azure commands
---
#use azure cloud shell to deploy, upload "env.yml, deployment.yml, endpoint.yml, score.py" to azure cloud shell
az account show --query name
az account set --subscription "put_subscription_name"
az account show --query name
az configure --defaults group="resource_group_name" workspace="azure_ML_workspace_name" location="eastasia"
az ml online-endpoint create -f endpoint.yml
az ml online-deployment create -f deployment.yml --all-traffic
az ml online-deployment update --name blue --endpoint-name put_your_endpoint_name -f deployment.yml

#take api url & key
az ml online-endpoint show -n distil-bert-final-model-endpoint --query scoring_uri -o tsv
az ml online-endpoint get-credentials -n distil-bert-final-model-endpoint --query primaryKey -o tsv

---

#for frontend 
#create azure static webapps and deploy from github use index.html file