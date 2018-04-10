# cs4225
CS4225 project: Automatic Personal Bank Transaction Extraction &amp; Categorization

## Requirements
- python 3

## AWS Lambda

PDF upload & parse

```
# local
LAMBDA_TASK_ROOT='/usr/local/' chalice local
curl localhost:8000/upload -F 'file=@path/to/file'

# remote
chalice deploy
curl https://42q6iw44o2.execute-api.ap-southeast-1.amazonaws.com/api/upload -F 'file=@path/to/file'
```
