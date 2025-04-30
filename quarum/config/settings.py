"""
Settings module for domain modeling.

This module provides a settings class to manage configuration
options and defaults for the domain modeling process.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class Settings:
    """
    Settings for domain modeling processes.
    
    This class manages configuration options, environment variables,
    and user preferences for the domain modeling framework.
    """
    
    # Default settings
    DEFAULT_SETTINGS = {
        # LLM settings
        "llm": {
            "model_name": "gpt-4.1-mini",
            "temperature": 0.0,
            "max_retries": 3,
            "retry_delay": 2.0,
            "timeout": 60
        },
        
        # Document processing
        "document": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "max_chunk_count": 100,
            "section_headers": [
                "# ", "## ", "### ", "Chapter ", "Section "
            ]
        },
        
        # Model generation
        "model": {
            "max_entities": 50,
            "max_relationships": 100,
            "confidence_threshold": 0.4,
            "banned_words": [
                "system", "component", "generic", "entity", 
                "item", "object", "module"
            ],
            "common_base_classes": [
                "Entity", "Resource", "Record", "Data", "Model", "Base"
            ]
        },
        
        # Output generation
        "output": {
            "diagram_style": "default",
            "include_metrics": True,
            "output_formats": ["plantuml", "markdown"],
            "output_directory": "output"
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize settings.
        
        Args:
            config_path: Optional path to a JSON config file
        """
        # Start with default settings
        self.settings = self.DEFAULT_SETTINGS.copy()
        
        # Load from config file if provided
        if config_path:
            self.load_from_file(config_path)
            
        # Override with environment variables
        self._load_from_env()
    
    def load_from_file(self, config_path: str) -> bool:
        """
        Load settings from a JSON config file.
        
        Args:
            config_path: Path to the config file
            
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            with open(config_path, 'r') as f:
                user_settings = json.load(f)
                
            # Update settings recursively
            self._update_dict_recursive(self.settings, user_settings)
            logger.info(f"Loaded settings from {config_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load settings from {config_path}: {str(e)}")
            return False
    
    def _update_dict_recursive(self, target: Dict, source: Dict) -> None:
        """
        Update a dictionary recursively.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively update nested dictionaries
                self._update_dict_recursive(target[key], value)
            else:
                # Set or override value
                target[key] = value
    
    def _load_from_env(self) -> None:
        """Load settings from environment variables."""
        # Look for environment variables with prefix QUARUM_
        prefix = "QUARUM_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Extract setting path from variable name
                setting_path = key[len(prefix):].lower().split('_')
                
                # Navigate to the appropriate setting
                current = self.settings
                for part in setting_path[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Set the value, converting to appropriate type
                setting_name = setting_path[-1]
                current[setting_name] = self._convert_value(value)
                
                logger.debug(f"Setting {key} from environment variable")
    
    def _convert_value(self, value: str) -> Any:
        """
        Convert string value to appropriate type.
        
        Args:
            value: String value to convert
            
        Returns:
            Converted value
        """
        # Try to convert to int
        try:
            return int(value)
        except ValueError:
            pass
            
        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            pass
            
        # Convert to boolean if applicable
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
            
        # Try to parse as JSON
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
            
        # Return as string
        return value
    
    def get(self, *path: str, default: Any = None) -> Any:
        """
        Get a setting value by path.
        
        Args:
            *path: Path components to the setting
            default: Default value if setting not found
            
        Returns:
            Setting value or default
        """
        current = self.settings
        
        for part in path:
            if part not in current:
                return default
            current = current[part]
            
        return current
    
    def set(self, *path_and_value: Any) -> None:
        """
        Set a setting value by path.
        
        Args:
            *path_and_value: Path components and value, where the last
                            element is the value to set
        """
        if len(path_and_value) < 2:
            logger.error("set() requires at least one path component and a value")
            return
            
        path = path_and_value[:-1]
        value = path_and_value[-1]
        
        current = self.settings
        
        # Navigate to the parent of the setting
        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
            
        # Set the value
        current[path[-1]] = value
    
    def save_to_file(self, config_path: str) -> bool:
        """
        Save current settings to a JSON file.
        
        Args:
            config_path: Path to save the config file
            
        Returns:
            True if successfully saved, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(self.settings, f, indent=2)
                
            logger.info(f"Saved settings to {config_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to save settings to {config_path}: {str(e)}")
            return False
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all settings.
        
        Returns:
            Copy of all settings
        """
        return self.settings.copy()
    
    def reset(self) -> None:
        """Reset all settings to defaults."""
        self.settings = self.DEFAULT_SETTINGS.copy()
        logger.info("Reset settings to defaults")
    
    def reset_section(self, section: str) -> bool:
        """
        Reset a specific section to defaults.
        
        Args:
            section: Section name
            
        Returns:
            True if section existed and was reset, False otherwise
        """
        if section in self.DEFAULT_SETTINGS:
            self.settings[section] = self.DEFAULT_SETTINGS[section].copy()
            logger.info(f"Reset {section} settings to defaults")
            return True
        else:
            logger.warning(f"Section {section} not found in default settings")
            return False