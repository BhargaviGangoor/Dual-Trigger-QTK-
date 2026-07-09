import os
import importlib.util
from typing import Dict, Any, List, Callable

class PluginManager:
    def __init__(self, plugins_dir: str = "backend/app/plugins"):
        self.plugins_dir = plugins_dir
        self.custom_attacks: Dict[str, Callable] = {}
        self.custom_models: Dict[str, Any] = {}
        self.custom_trust_rules: Dict[str, Callable] = {}

    def discover_plugins(self):
        """Discovers and dynamically imports custom Python modules in the plugins directory."""
        if not os.path.exists(self.plugins_dir):
            os.makedirs(self.plugins_dir, exist_ok=True)
            return

        for filename in os.listdir(self.plugins_dir):
            if filename.endswith(".py") and filename != "__init__.py" and filename != "plugin.py":
                filepath = os.path.join(self.plugins_dir, filename)
                plugin_name = filename[:-3]
                
                try:
                    spec = importlib.util.spec_from_file_location(plugin_name, filepath)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Register custom attacks
                        if hasattr(module, "register_attack"):
                            name, func = module.register_attack()
                            self.custom_attacks[name] = func
                            
                        # Register custom ML models
                        if hasattr(module, "register_model"):
                            name, model = module.register_model()
                            self.custom_models[name] = model
                            
                        # Register custom trust policies
                        if hasattr(module, "register_trust_rule"):
                            name, rule = module.register_trust_rule()
                            self.custom_trust_rules[name] = rule
                            
                except Exception as e:
                    print(f"Error loading plugin {plugin_name} from {filepath}: {e}")

    def get_attacks_list(self) -> List[str]:
        return list(self.custom_attacks.keys())

    def get_models_list(self) -> List[str]:
        return list(self.custom_models.keys())
