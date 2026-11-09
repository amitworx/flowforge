"""
Custom upload handler to enforce maximum file upload sizes at the streaming boundary.
Protects application memory from unbounded multipart payloads before parsing.
"""

from django.core.files.uploadhandler import FileUploadHandler, StopUpload

from flowforge.processing.services import MAX_UPLOAD_SIZE_BYTES


class BoundedFileUploadHandler(FileUploadHandler):
    """
    Enforces a strict upper bound on uploaded files during streaming chunks.
    Raises StopUpload(connection_reset=False) if the uploaded file exceeds max_file_size.
    """

    def __init__(self, request=None, max_file_size: int = MAX_UPLOAD_SIZE_BYTES):
        super().__init__(request)
        self.max_file_size = max_file_size
        self.total_upload_size = 0
        self.activated = False
        self.upload_aborted = False

    def receive_data_chunk(self, raw_data: bytes, start: int) -> bytes:  # noqa: ARG002
        self.activated = True
        self.total_upload_size += len(raw_data)
        if self.total_upload_size > self.max_file_size:
            self.upload_aborted = True
            # Signal Django's multipart parser to halt further processing of this upload
            raise StopUpload(connection_reset=False)
        return raw_data

    def file_complete(self, file_size: int):  # noqa: ARG002
        return None
