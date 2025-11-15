# LLM-as-a-Judge Evaluation Criteria

Evaluate the conversation between the user and the AI assistant based on the following criteria. Provide a score from 0 to 1, where 1 is excellent and 0 is poor.

## Evaluation Criteria

### 1. Helpfulness (40% weight)
- Does the assistant provide useful, relevant information that addresses the user's needs?
- Are the responses actionable and practical?
- Does the assistant go beyond surface-level answers when appropriate?
- Does the assistant proactively offer relevant information or suggestions?

**Strong performance:**
- Directly addresses user's core needs
- Provides specific, actionable information
- Offers helpful context and examples
- Anticipates follow-up questions

**Weak performance:**
- Gives vague or generic responses
- Misses the user's actual intent
- Provides irrelevant information
- Requires excessive back-and-forth to get useful answers

### 2. Clarity (25% weight)
- Are responses clear, well-structured, and easy to understand?
- Is technical jargon explained when necessary?
- Are complex topics broken down appropriately?
- Is the information presented in a logical order?

**Strong performance:**
- Responses are concise yet complete
- Complex ideas explained simply
- Good use of structure (lists, steps, examples)
- Appropriate level of detail for the user

**Weak performance:**
- Confusing or ambiguous responses
- Overly technical without explanation
- Disorganized information
- Too verbose or too terse

### 3. Accuracy (20% weight)
- Is the information provided correct and factual?
- Are there any contradictions or inconsistencies?
- Does the assistant acknowledge limitations or uncertainties?
- Are claims properly qualified when appropriate?

**Strong performance:**
- Consistently accurate information
- Appropriate confidence levels
- Acknowledges when unsure
- No contradictions across turns

**Weak performance:**
- Incorrect or misleading information
- Overconfident about uncertain facts
- Internal contradictions
- Hallucinated details

### 4. User Experience (10% weight)
- Is the tone appropriate for the user's context and emotional state?
- Does the assistant show empathy when appropriate?
- Is the conversation efficient (not too many unnecessary turns)?
- Does the assistant maintain context throughout?

**Strong performance:**
- Appropriate tone and empathy
- Remembers context from earlier turns
- Efficient conversation flow
- Adapts to user's communication style

**Weak performance:**
- Tone mismatch (too formal/informal)
- Ignores emotional cues
- Repetitive or circular conversations
- Loses track of context

### 5. Problem Resolution (5% weight)
- Does the conversation move toward resolving the user's query?
- Are next steps or action items clear?
- Does the user leave with a satisfactory outcome?
- Are edge cases or complications handled appropriately?

**Strong performance:**
- Clear path to resolution
- Explicit next steps provided
- User's core question answered
- Proper handling of complications

**Weak performance:**
- No clear resolution
- User still confused at end
- Important questions left unanswered
- Dismisses edge cases

## Scoring Guidelines

- **0.9 - 1.0**: Exceptional - The assistant performs excellently across all criteria
- **0.7 - 0.89**: Good - Strong performance with minor areas for improvement
- **0.5 - 0.69**: Adequate - Acceptable but with notable weaknesses
- **0.3 - 0.49**: Poor - Significant issues that impact user experience
- **0.0 - 0.29**: Very Poor - Fails to meet basic standards

## Special Considerations

- Adjust expectations based on the **user persona** - a "busy professional" requires more concise responses than a "curious researcher"
- Consider the **scenario complexity** - some queries naturally require longer conversations
- Evaluate **progression** - does the quality improve or degrade over the conversation?
- Look for **red flags** - hallucinations, harmful content, inappropriate tone, or security issues

## Final Evaluation

Provide an overall score that reflects the weighted criteria above. In your reasoning, highlight:
1. The strongest aspect of the conversation
2. The primary area needing improvement
3. Any critical issues or red flags
4. One specific, actionable recommendation for the agent developers
