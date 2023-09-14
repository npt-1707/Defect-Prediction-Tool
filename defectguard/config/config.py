from importlib.resources import files

# Get the absolute path of the file
absolute_path = str(files('config').joinpath('colab-385406-e848129cc804.json'))

# Print the absolute path
print("Absolute path:", absolute_path)

CONFIG = {
    'remote': {
        'storage': {
            'gdrive_client_id': '515220301377-6015nql03h01qkg5maahiusa1int0b54.apps.googleusercontent.com',
            'gdrive_client_secret': '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',
            'gdrive_use_service_account': True,
            'gdrive_service_account_user_email': 'defectguard@colab-385406.iam.gserviceaccount.com',
            'gdrive_service_account_json_file_path': absolute_path,
        },
    },
}