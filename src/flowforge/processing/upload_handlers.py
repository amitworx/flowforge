"""
Custom upload handler to enforce maximum file upload sizes at the streaming boundary.
Protects application memory from unbounded multipart payloads before parsing.
"""

from django.core.files.uploadhandler import FileUploadHandler, StopUpload


class BoundedFileUploadHandler(FileUploadHandler):
    """
    Enforces a strict upper bound on uploaded files during streaming chunks.
    Raises StopUpload(connection_reset=False) if the uploaded file exceeds max_file_size.
    """

    def __init__(self, request=None, max_file_size: int = 1024 * 1024):
        super().__init__(request)
        self.max_file_size = max_file_size
        self.total_upload_size = 0

    def receive_data_chunk(self, raw_data: bytes, start: int) -> bytes:  # noqa: ARG002
        self.total_upload_size += len(raw_data)
        if self.total_upload_size > self.max_file_size:
            # Signal Django's multipart parser to halt further processing of this upload
            raise StopUpload(connection_reset=False)
        return raw_data

    def file_complete(self, file_size: int):  # noqa: ARG002
        return None
