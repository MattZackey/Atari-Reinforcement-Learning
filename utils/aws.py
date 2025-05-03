import logging
import os
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

def check_s3_bucket(bucket_name):
    
    s3 = boto3.client('s3')
    try:
        s3.head_bucket(Bucket = bucket_name)
        logger.info(f'Using S3 Bucket: {bucket_name}')
    except ClientError as e:
        logger.error(f'Error S3: {e}')
        raise
    except Exception as ex: 
        logger.error(f"An unexpected error occurred for S3: {ex}")
        raise
        
def create_s3_keys(bucket_name, save_root_folder, game_name):
    
    s3 = boto3.client('s3')
    
    # Create Agent key on S3
    try:
        s3.head_object(Bucket=bucket_name, Key=f"{save_root_folder}/{game_name}/agent/")
    except:
        logger.info("Creating S3 key to save agent")
        s3.put_object(Bucket=bucket_name, Key=f"{save_root_folder}/{game_name}/agent/")
        
    # Create game info key on S3
    try:
        s3.head_object(Bucket=bucket_name, Key=f"{save_root_folder}/{game_name}/game/")
    except:
        logger.info("Creating S3 key to save game info")
        s3.put_object(Bucket=bucket_name, Key=f"{save_root_folder}/{game_name}/game/")
        
    # Create gameplay key on S3
    try:
        s3.head_object(Bucket=bucket_name, Key=f"{save_root_folder}/{game_name}/gameplay/")
    except:
        logger.info("Creating S3 key to save gameplay")
        s3.put_object(Bucket=bucket_name, Key=f"{save_root_folder}/{game_name}/gameplay/")
        
    # Create lots key on S3
    try:
        s3.head_object(Bucket=bucket_name, Key=f"{save_root_folder}/{game_name}/plots/")
    except:
        logger.info("Creating S3 key to save plots")
        s3.put_object(Bucket=bucket_name, Key=f"{save_root_folder}/{game_name}/plots/")
        
    # Create local directories
    if not os.path.exists(f"/opt/ml/output/{game_name}"):
        os.makedirs(f"/opt/ml/output/{game_name}/agent/")
        os.makedirs(f"/opt/ml/output/{game_name}/game/")
        os.makedirs(f"/opt/ml/output/{game_name}/gameplay/")
        os.makedirs(f"/opt/ml/output/{game_name}/plots/")