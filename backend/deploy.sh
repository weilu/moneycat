#!/bin/bash
set -ex

# https://rsalveti.wordpress.com/2007/04/03/bash-parsing-arguments-with-getopts/
usage()
{
cat << EOF
usage: $0 options

This packages & deploy the chalice app to aws labmda

OPTIONS:
   -h      Show this message
   -r      Rebuild python dependencies
EOF
}

REBUILD_DEP=0
while getopts “hr” OPTION
do
  case $OPTION in
    h)
      usage
      exit 1
      ;;
    r)
      REBUILD_DEP=1
      ;;
    ?)
      usage
      exit
      ;;
  esac
done

if [ $REBUILD_DEP -eq 1 ]
then
  cd sklearn-build-lambda
  docker run -v $(pwd):/outputs -it amazonlinux /bin/bash /outputs/build.sh
  mv venv.zip ../vendor/
  cd ../vendor
  unzip -o venv.zip
  rm venv.zip
  cd ../
  echo "vendor folder uncompressed size $(du -sh ./vendor | cut -f1)"
  echo "It needs to be less than 250MB to be deployable"
fi

chalice package .chalice/deployments/
aws s3 cp .chalice/deployments/deployment.zip s3://cs4225-models/
aws lambda update-function-code --function-name parseBankStatement-dev --region ap-southeast-1 --s3-bucket cs4225-models --s3-key deployment.zip
