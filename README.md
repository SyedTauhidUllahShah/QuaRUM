# Domain Modeler

A research-oriented framework for automatically extracting UML domain models from natural language requirements using AI and qualitative coding techniques.

## Overview

Domain Modeler applies qualitative data analysis techniques from grounded theory (open coding, axial coding, and selective coding) in combination with large language models to analyze requirements documents and produce coherent domain models.

The framework extracts:
- Classes, interfaces, and enumerations
- Attributes and methods
- Relationships and hierarchies
- Actors and their interactions

## Features

- **Research-Oriented**: Modular design, instrumentation for metrics, extensible architecture
- **AI-Powered**: Uses LLMs to extract entities and relationships from text
- **Qualitative Analysis**: Follows methodical coding approach from grounded theory
- **Rich Outputs**: Generates PlantUML diagrams and detailed traceability reports
- **Customizable**: Configurable settings for adapting to different domains

## Installation

```bash
# Clone the repository
git clone https://github.com/example/domain_modeler.git
cd domain_modeler

# Install the package
pip install -e .
```

## Requirements

- Python 3.8+
- OpenAI API key

## Usage

### Command-line Interface

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your-api-key

# Run domain modeling on a requirements file
domain-modeler --file path/to/requirements.txt --description "E-commerce system for managing products, orders, and customers"
```

### Python API

```python
from domain_modeler.pipelines.domain_modeling import DomainModelingPipeline

# Initialize pipeline
pipeline = DomainModelingPipeline(api_key="your-openai-key")

# Execute pipeline
result = pipeline.execute(
    file_path="requirements.txt",
    domain_description="E-commerce system for selling products",
    output_dir="models"
)

# Access results
print(f"Success: {result.success}")
print(f"Entities: {result.metrics['final_entity_count']}")
print(f"Relationships: {result.metrics['final_relationship_count']}")
print(f"UML diagram saved to: {result.outputs['plantuml_file']}")
```

### Example Script

```bash
python examples/simple_modeling.py --file requirements.txt --domain "E-commerce system" --api-key your-openai-key
```

## Architecture

The framework is organized as a pipeline of distinct research phases:

1. **Open Coding**: Extract entities from text segments
2. **Axial Coding**: Identify relationships between entities
3. **Selective Coding**: Refine the model, establish hierarchies, enhance details
4. **Validation**: Ensure consistency and correctness
5. **Generation**: Produce UML diagrams and traceability reports

## Customization

Settings can be customized through the Settings class:

```python
from domain_modeler.config.settings import Settings

settings = Settings()
settings.set("llm", "model_name", "gpt-4")
settings.set("output", "diagram_style", "vibrant")
```

## License

MIT License