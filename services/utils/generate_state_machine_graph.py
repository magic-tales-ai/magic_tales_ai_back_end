from graphviz import Digraph

dot = Digraph(comment='MagicTales.ai Story Generation Workflow')

# Define states
states = {
    'Start': 'Start',
    'Data_Extraction': 'Data Extraction',
    'Story_Generation': 'Story Generation',
    'Image_Prompt_Generation': 'Image Prompt Generation',
    'Image_Generation': 'Image Generation',
    'Document_Generation': 'Document Generation',
    'Error': 'Error',
    'End': 'End'
}

# Add states to the graph
for state_id, label in states.items():
    dot.node(state_id, label)

# Define transitions with labels
transitions = [
    ('Start', 'Data_Extraction', 'User logs in'),
    ('Data_Extraction', 'Story_Generation', '(Profile, Story Features, Synopsis)'),
    ('Story_Generation', 'Image_Prompt_Generation', '(Story Chapters)'),
    ('Story_Generation', 'Error', ''),
    ('Image_Prompt_Generation', 'Image_Generation', '(Image Prompts)'),
    ('Image_Prompt_Generation', 'Error', ''),
    ('Image_Generation', 'Document_Generation', '(Images)'),
    ('Image_Generation', 'Error', ''),
    ('Document_Generation', 'End', '(PDF)'),
    ('Document_Generation', 'Error', ''),
    ('Error', 'Data_Extraction', '(Re-Login)')
]

# Add transitions to the graph
for start, end, label in transitions:
    dot.edge(start, end, label=label)

# Generate and save the diagram
logger.info(dot.source)  # Print the DOT language code (optional)
dot.render('magic_tales_workflow', format='png')  # Render as PNG image
