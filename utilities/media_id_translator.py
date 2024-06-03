import json
import os
import threading

from config.config_manager import ConfigManager


class media_id_translator():
    """Translate media id into a url
    TODO: replace with sql database or something else, This is simple but wont scale...
    """

    def __init__(self, root_path="", translation_file="./translator.json"):
        """Initialize object

        Args:
            root_path (str, optional): Root path of media. Defaults to "".
            translation_file (str, optional): Translation file associated. Defaults to "./translator.json".
        """
        self.root_path=root_path
        self.lock = threading.Lock()
        self.default_translation_file=json.loads(ConfigManager()["spectrogram"]["default_translation"])
        self.translation_file = translation_file
        self.translator_id_img=None

    def create_new_access(self, rel_filepath="", id_guess=None):
        """Generate an access key for an image

        Args:
            rel_filepath (str, optional): file path to look. Defaults to "".
            id_guess (int, optional): guess of a preferred ID. Defaults to None.

        Returns:
            int: ID
        """
        with self.lock:
            if(os.path.exists(os.path.join(self.root_path,self.translation_file))):
                file_contents=open(os.path.join(self.root_path,self.translation_file),"r").read()
                if file_contents!="":
                    try:
                        self.translator_id_img=json.loads(file_contents)
                    except Exception as e:
                        open(os.path.join(self.root_path,self.translation_file),"w").write(json.dumps(self.default_translation_file))
                        print(e)
                        print("Catastrophic failure media id translator. Saving failed contents")
                        # open(os.path.exists(os.path.join(root_path,translation_file.replace(".json","_failed.json"))),"w").write(file_contents)
                else:
                    self.translator_id_img=self.default_translation_file
            else:
                self.translator_id_img=self.default_translation_file
            if rel_filepath not in list(self.translator_id_img.values()):
                if rel_filepath!=None and id_guess==None:
                    id_guess = len(list(self.translator_id_img.keys()))
                    while id_guess in list(self.translator_id_img.keys()):
                        id_guess+=1
                    #found new id
                    self.translator_id_img[id_guess]=rel_filepath
                    out = json.dumps(self.translator_id_img)
                    open(os.path.join(self.root_path,self.translation_file),"w").write(out)
                    return id_guess
            else:
                id_guess = [i for i in self.translator_id_img.keys() if self.translator_id_img[i]==rel_filepath]
                return id_guess[0]

    def __getitem__(self, key):
        """Return image path given an ID

        Args:
            key (int): ID

        Returns:
            str: Path
        """
        with self.lock:
            self.translator_id_img=self.default_translation_file
            if(os.path.exists(os.path.join(self.root_path,self.translation_file))):
                file_contents=open(os.path.join(self.root_path,self.translation_file),"r").read()
                if file_contents!="":
                    try:
                        self.translator_id_img=json.loads(file_contents)
                    except Exception as e:
                        print(e)
                        print("Catastrophic failure media id translator. Saving failed contents")
                        # open(os.path.exists(os.path.join(root_path,translation_file.replace(".json","_failed.json"))),"w").write(file_contents)
                else:
                    self.translator_id_img=self.default_translation_file
            else:
                self.translator_id_img=self.default_translation_file
            return self.translator_id_img[key]
    

if __name__ == "__main__":
    import random
    import time

    for x in range(0,10000):
        start_time=time.time()
        media_id_translator().create_new_access(rel_filepath=str(random.randint(0,100000000)))
        end_time=time.time()
        # print(f"took :{end_time-start_time}s")