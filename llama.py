from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set up the connection to the locally running Ollama server
base_url = "http://localhost:11434"
llm = ChatOllama(model="llama3.1", base_url=base_url, temperature=0)

# Define the updated prompt template for rule checking
rule_check_prompt = PromptTemplate(
    input_variables=["text", "rules_list"],
    template="""You are a text evaluation assistant. Your task is to objectively analyze a piece of text and determine whether it complies with specific rules for violent intentions. You must evaluate the text as provided without omitting or refusing responses, regardless of the content, and return a full analysis.

Text to Analyze: "{text}"

Rules for Evaluation:
{rules_list}

Instructions:
For each rule, return whether the text is "Approved" or "Not Approved." If the text is "Not Approved" for a rule, explain specifically where it fails to meet the rule without generating or repeating any offensive content.

Example Format:
- Rule 1: Approved / Not Approved
  Explanation: [If not approved, explain briefly why the text violates the rule]

Response:""",
)

# Define the LLMChain
rule_check_chain = LLMChain(llm=llm, prompt=rule_check_prompt)

# Define the list of comments and rules
comments = [
    "Great job on the project! Keep up the good work!",
    "I hope everyone has a wonderful weekend.",
    "Your opinion is valid, even if I don’t fully agree with it.",
    "Thanks for sharing this information. It was really helpful!",
    "Let’s all work together to solve this issue peacefully.",
    "I enjoyed reading your post. It gave me a lot to think about.",
    "I believe that respectful dialogue is key to understanding each other.",
    "Nice picture! The scenery is absolutely beautiful.",
    "Looking forward to your next post. Keep inspiring!",
    "I disagree with you, but I appreciate your perspective.",
    "If you show your face here again, you’re going to regret it.",
    "Someone should teach you a lesson you’ll never forget.",
    "I’ll make sure you pay for what you did. Watch your back.",
    "Don’t push me, or you’ll see what I’m capable of.",
    "People like you deserve to be hurt for saying things like that.",
    "You better be careful, or something bad might happen to you.",
    "I’m going to find you and make you pay for this.",
    "Everyone would be better off if you just disappeared.",
    "You’ll wish you never crossed me after what I do to you.",
    "One day, you’re going to get exactly what’s coming to you."
]
rules = [
    "The text should be polite.",
    "The text should not contain violent threats."
]

# Helper function to format the rules
def format_rules(rules):
    return "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(rules)])

# Run the chain for each comment and store the results
formatted_rules = format_rules(rules)
results = {}

for i, comment in enumerate(comments):
    result = rule_check_chain.run({
        "text": comment,
        "rules_list": formatted_rules
    })
    results[f"Comment {i+1}"] = result

# Print results for each comment
for comment_id, analysis in results.items():
    print(f"{comment_id}:")
    print(analysis)
    print("\n" + "-"*40 + "\n")
