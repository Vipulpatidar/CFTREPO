#!/bin/bash

# Set variables
BUCKET_NAME="t"
AWS_REGION=${AWS_REGION:-"us-east-1"}  # Default to us-east-1 if not set
FILES_DIR="lambda_code"  # Directory containing the files to upload

# Create S3 bucket
echo "Creating S3 bucket: $BUCKET_NAME in region: $AWS_REGION"
aws s3 mb s3://$BUCKET_NAME --region $AWS_REGION

# Verify bucket creation
if [ $? -ne 0 ]; then
  echo "Failed to create S3 bucket!"
  exit 1
fi

# Upload files to S3
echo "Uploading files from $FILES_DIR to S3 bucket..."
aws s3 cp $FILES_DIR s3://$BUCKET_NAME/ --recursive

# Verify file upload
if [ $? -ne 0 ]; then
  echo "Failed to upload files!"
  exit 1
fi

echo "S3 bucket and file upload completed successfully!"
