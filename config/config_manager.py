import toml


class ConfigManager():
    """Manage the default config
    """
    def __init__(self):
        """Initialize the config
        """
        self.__config = self.parse_config()
    def __get_config_path(self):
        return "./config/default.toml"
    def parse_config(self):
        """Parse the toml config
        Returns:
            dict: Config dict
        """
        file_path = self.__get_config_path()
        with open(file_path, "r") as file:
            config = toml.load(file_path)
        return config
    def __getitem__(self, key):
        """[] class overload

        Args:
            key (str): Key

        Returns:
            str: Value
        """
        return self.__config[key]
