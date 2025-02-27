import os
import time
from itertools import product

class BlockingFileList:
    """
    A list of files in a directory with blocking retrieval: if the file does not exist, wait upto a given timeout for it to appear.
    This is useful if input files for a workflow may be generated by an asyncronous process, and we therefore cannot guarantee that 
    they will have been created beforehand.

    To support the fact that the exact file names may not be known ahead of time, you may specify a list of possible extensions and
    an optional prefix. Additionally, the desired index may be padded with zeros to a fixed length of 9, or not.
    
    For example, for a given access like `blockingFileList[34]` where optional_prefix="foo" and extensions=["jpg", "png"], the order
    in which the files are checked is:
    - "foo000000034.jpg"
    - "foo000000034.png"
    - "foo34.jpg"
    - "foo34.png"
    - "000000034.jpg"
    - "000000034.png"
    - "34.jpg"
    - "34.png"

    Attributes:
        base_directory (str): The directory where the files are located.
        expected_file_count (int): The expected number of files that will constitute this list. This is required because client code 
        may check for the length of the list before all files are generated.
        optional_prefix (str): An optional prefix to prepend to the file names.
        extensions (list of str): The list of file extensions to search for.
        timeout_seconds (int): The maximum time (in seconds) to wait for a file to appear.
    """    
 
    def __init__(self, base_directory, expected_file_count, optional_prefix="", extensions=["jpg", "png"], timeout_seconds=30):
        self.base_directory = base_directory
        self.expected_file_count = expected_file_count
        self.extensions = extensions
        self.timeout_seconds = timeout_seconds
        self.optional_prefix = optional_prefix

    def __getitem__(self, index) -> str:
        """
        Raises:
            FileNotFoundError: If no file matching the criteria is found within the timeout period.
        """        
        start_time = time.time()

        waited = 0
        while waited < self.timeout_seconds:
            for prefix, number, ext in product([self.optional_prefix, ""], [f"{index:09}", f"{index}"], self.extensions):
                file_path = os.path.join(self.base_directory, f"{prefix}{number}.{ext}")
                if os.path.exists(file_path):
                    return file_path

            waited = time.time() - start_time
            print(f"Could not find matching {self.extensions} file for index {index} in {self.base_directory}. Waited {waited:.2f}/{self.timeout_seconds}s...")
            time.sleep(1)  # Wait for 1 second before checking again
        
        raise FileNotFoundError(f"No file with matching {self.extensions} file for index {index} in {self.base_directory} after waiting for {self.timeout_seconds} seconds")

    def __len__(self) -> int:
        return self.expected_file_count