import boto3

sns_client = boto3.client('sns', region_name='us-west-2')

def send_sms_alert(message):
    phone_number = "+9999999"  
    sns_client.publish(PhoneNumber=phone_number, Message=message)
