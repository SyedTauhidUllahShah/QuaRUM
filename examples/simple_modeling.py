"""
Simple domain modeling example.

This example demonstrates how to use the domain_modeler package
to extract a domain model from requirements text.
"""

import os
import argparse
from domain_modeler.pipelines.domain_modeling import DomainModelingPipeline
from domain_modeler.config.settings import Settings

def main():
    """Run the example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Domain Modeling Example")
    
    parser.add_argument(
        "--file", 
        type=str, 
        required=True,
        help="Path to the requirements text file"
    )
    
    parser.add_argument(
        "--api-key", 
        type=str, 
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key (defaults to OPENAI_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--domain", 
        type=str, 
        required=True,
        help="Domain description (brief summary of the domain)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output",
        help="Directory to save output files"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4.1-mini",
        help="OpenAI model to use"
    )
    
    args = parser.parse_args()
    
    # Validate API key
    if not args.api_key:
        raise ValueError(
            "OpenAI API key is required. Provide it with --api-key or "
            "set the OPENAI_API_KEY environment variable."
        )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create custom settings
    settings = Settings()
    settings.set("llm", "model_name", args.model)
    settings.set("output", "diagram_style", "vibrant")  # Use vibrant style for the diagram
    
    # Initialize pipeline
    pipeline = DomainModelingPipeline(
        api_key=args.api_key,
        model_name=args.model
    )
    
    # Print starting message
    print(f"Starting domain modeling for: {args.domain}")
    print(f"Using file: {args.file}")
    print(f"Using model: {args.model}")
    
    # Execute pipeline
    result = pipeline.execute(
        file_path=args.file,
        domain_description=args.domain,
        output_dir=args.output_dir
    )
    
    # Print results
    if result.success:
        print("\n✅ Domain modeling completed successfully!")
        print(f"Entities extracted: {result.metrics.get('final_entity_count', 0)}")
        print(f"Relationships identified: {result.metrics.get('final_relationship_count', 0)}")
        print(f"Execution time: {result.execution_time:.2f} seconds")
        print("\nOutput files:")
        print(f"- PlantUML diagram: {result.outputs.get('plantuml_file')}")
        print(f"- Traceability report: {result.outputs.get('report_file')}")
        
        # Suggest next steps
        print("\nNext steps:")
        print("1. View the PlantUML diagram at http://www.plantuml.com/plantuml/uml/")
        print("2. Review the traceability report for details on extracted entities")
        print("3. Iterate by refining the requirements or domain description")
    else:
        print("\n❌ Domain modeling failed.")
        print("Check the error messages below:")
        for message in result.messages:
            if message.startswith("ERROR"):
                print(f"  {message}")

if __name__ == "__main__":
    main()