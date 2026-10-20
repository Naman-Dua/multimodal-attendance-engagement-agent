class InsightGenerator:
    def generate_classroom_insight(self, session_data: dict) -> dict:
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert educational AI assistant. Provide brief, professional, and actionable insights for",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="mixtral-8x7b-32768",
                temperature=0.5,
                max_tokens=150,
            )

            response_text = chat_completion.choices[0].message.content

            # Simple Parsing
            insight = "Class status is stable."
            recommendation = "Maintain current pace."
