custom_prompt = PromptTemplate(
    """\
Rewrite the user's follow-up question as a standalone question.

1. Include all relevant past context.
2. Keep it natural and grammatically correct.
3. If already standalone, return it unchanged.

<Chat History>
{chat_history}

<User's Follow-Up Question>
{question}

<Rewritten Standalone Question>
"""
)


response_prompt = PromptTemplate(
    """\
You are an AI assistant providing structured responses.

### **Instructions:**
- Answer clearly and concisely.
- Summarize retrieved context to avoid duplication.
- Summarize the key facts efficiently.
- If the context lacks enough details, say: "I don’t have enough information."
- Format responses in natural sentences.

<Retrieved Context>
{context}

<User's Query>
{question}

### **AI Response:**
"""
)