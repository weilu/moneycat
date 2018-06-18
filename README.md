# cs4225
CS4225 project: Automatic Personal Bank Transaction Extraction &amp; Categorization

## Requirements
- python 3
- docker (for deployment binary building)
- aws cli (for deployment)

## AWS Lambda

API Usage

```
# PDF upload, parse & classification, returns a csv or json, auth required
# Use -H "Accept: application/json" header to control response format
curl https://q5i6ef1jfi.execute-api.ap-southeast-1.amazonaws.com/api/upload -F 'file=@[path/to/file].pdf' -F 'password=[pdf password]' -H 'Authorization: [jwtToken]'

# CSV upload, after user reviews & confirms parse & classification results, auth required
curl -v https://q5i6ef1jfi.execute-api.ap-southeast-1.amazonaws.com/api/confirm -X POST --data-urlencode 'file@[path/to/file].csv' -H 'Authorization: [jwtToken]'

# User transaction data fetch, returns a csv or json of all transaction data for the given user, auth required
# Use -H "Accept: application/json" header to control response format
curl https://q5i6ef1jfi.execute-api.ap-southeast-1.amazonaws.com/api/transactions -H 'Authorization: [jwtToken]'

# Refresh model with new csv data
curl -v https://q5i6ef1jfi.execute-api.ap-southeast-1.amazonaws.com/api/refresh-model
```

Development

```
cd backend

# local
LAMBDA_TASK_ROOT='/usr/local/' chalice local
curl localhost:8000/upload -F 'file=@path/to/file'

# deployment, requires aws cli setup
bash deploy.sh

# rebuild python dependencies & redeploy, requires aws cli + docker
git submodule init
git submodule update
bash deploy.sh -r
```

Be careful about add dependencies to backend (labmda) as the packaged code size limits are as follow:

- zip: no more than 50MB when upload directly, no more than 250MB when upload to s3 first & deploy to lambda from s3
- unzipped: no more than 250MB

After adding new dependency, you will need to deploy with the `-r` option

## TODO

- Handle missing labels/categories

## References

https://hackernoon.com/exploring-the-aws-lambda-deployment-limits-9a8384b0bec3
https://serverlesscode.com/post/deploy-scikitlearn-on-lamba/
http://chalice.readthedocs.io/en/latest/topics/packaging.html
http://chalice.readthedocs.io/en/latest/topics/views.html#binary-content
https://github.com/skylander86/lambda-text-extractor/blob/master/functions/simple/main.py
