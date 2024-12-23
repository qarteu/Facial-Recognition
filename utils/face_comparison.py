from utils.aws_client import rekognition_client

def compare_faces(source_image_bytes, target_image_bytes):
    response = rekognition_client.compare_faces(
        SourceImage={'Bytes': source_image_bytes},
        TargetImage={'Bytes': target_image_bytes},
        SimilarityThreshold=90
    )
    return len(response['FaceMatches']) > 0
