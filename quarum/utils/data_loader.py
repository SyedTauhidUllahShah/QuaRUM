"""
Dataset loading utilities for domain modeling.
"""

import os


class DataLoader:
    """Loads datasets for domain modeling."""

    def __init__(self, data_root: str = "data"):
        """
        Initialize the data loader.

        Args:
            data_root: Root directory for datasets
        """
        self.data_root = data_root

    def get_requirements_path(self, domain: str, filename: str) -> str:
        """
        Get path to a requirements file.

        Args:
            domain: Domain name (e.g., "ecommerce")
            filename: Requirements filename

        Returns:
            Full path to the requirements file
        """
        return os.path.join(self.data_root, "requirements", domain, filename)

    def list_available_domains(self) -> list[str]:
        """
        list available domains with requirements.

        Returns:
            list of domain names
        """
        domains_dir = os.path.join(self.data_root, "requirements")
        if not os.path.exists(domains_dir):
            return []

        return [
            d
            for d in os.listdir(domains_dir)
            if os.path.isdir(os.path.join(domains_dir, d))
        ]

    def list_domain_files(self, domain: str) -> list[str]:
        """
        list requirement files for a domain.

        Args:
            domain: Domain name

        Returns:
            list of requirement filenames
        """
        domain_dir = os.path.join(self.data_root, "requirements", domain)
        if not os.path.exists(domain_dir):
            return []

        return [
            f
            for f in os.listdir(domain_dir)
            if os.path.isfile(os.path.join(domain_dir, f))
        ]
