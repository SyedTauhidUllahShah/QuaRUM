"""
Command-line interface for the Domain Modeler tool.

This module provides the entry point for command-line usage of the
domain modeling framework, allowing users to process requirements
files and generate UML models.
"""

import os
import argparse
from quarum.pipelines.domain_modeling import DomainModelingPipeline

def main():
    """Main entry point for the domain modeling command-line interface."""
    parser = argparse.ArgumentParser(
        description="Domain Modeler: Extract UML models from natural language requirements"
    )
    
    parser.add_argument(
        "--file", 
        type=str, 
        required=True, 
        help="Path to the requirements file"
    )
    
    parser.add_argument(
        "--description", 
        type=str, 
        required=True, 
        help="Domain description"
    )
    
    parser.add_argument(
        "--api-key", 
        type=str, 
        required=True, 
        help="OpenAI API key"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output", 
        help="Output directory"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4.1-mini", 
        help="LLM model name"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize and run the pipeline
    pipeline = DomainModelingPipeline(
        api_key=args.api_key,
        model_name=args.model
    )
    
    pipeline.execute(
        file_path=args.file,
        domain_description=args.description,
        output_dir=args.output_dir
    )
    
    return 0

if __name__ == "__main__":
    exit(main())