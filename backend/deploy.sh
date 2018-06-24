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
   -s      stage to deploy to, can be dev (default) or prod
EOF
}

REBUILD_DEP=0
STAGE=dev
while getopts “hrs:” OPTION
do
  case $OPTION in
    h)
      usage
      exit 1
      ;;
    r)
      REBUILD_DEP=1
      ;;
    s)
      STAGE=$OPTARG
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

# package chalice & upload bundle to s3
chalice package --stage $STAGE .chalice/deployments/
aws s3 cp .chalice/deployments/deployment.zip s3://moneycat-deployment-$STAGE/

mv vendor vendor_tmp # workaround: such that chalice deploy won't complain about package size
chalice deploy --stage $STAGE # make sure aws API gateway is re-deployed
mv vendor_tmp vendor

# update lambda code from s3
aws lambda update-function-code --function-name money-api-$STAGE --region ap-southeast-1 --s3-bucket moneycat-deployment-$STAGE --s3-key deployment.zip
