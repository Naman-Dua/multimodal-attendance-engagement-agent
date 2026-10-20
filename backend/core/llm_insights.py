import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class InsightGenerator:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        try:
            self.client = Groq(api_key=self.api_key) if self.api_key else None
        except Exception:
            self.client = None

    def generate_classroom_insight(self, session_data: dict) -> dict:
        prompt = (
            f"Classroom Session Summary:\n"
            f"- Active Participants: {session_data.get('participants_count', 0)}\n"
            f"- Average Engagement Score: {session_data.get('average_engagement', 0.0):.2f}\n"
            f"- Noise / Audio Activity Level: {session_data.get('audio_activity', 'normal')}\n"
            f"- Attention Drops Detected: {session_data.get('attention_drops', 0)}\n"
            f"- Pacing Indicator: {session_data.get('pacing', 'steady')}\n\n"
            f"Provide a brief evaluation of current class engagement and one actionable teaching recommendation."
        )

        try:
            if not self.client:
                raise ValueError("Groq client not initialized (missing API key).")

            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert educational AI assistant. Provide brief, professional, and actionable insights for educators based on classroom engagement metrics.",
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

            lines = [line.strip() for line in response_text.strip().split("\n") if line.strip()]
            if len(lines) >= 2:
                insight = lines[0].replace("Insight:", "").strip()
                recommendation = lines[1].replace("Recommendation:", "").strip()
            elif len(lines) == 1:
                insight = lines[0]

            return {
                "insight": insight,
                "recommendation": recommendation,
                "raw_response": response_text
            }
        except Exception as e:
            # Rule-based fallback if offline or API unavailable
            avg_eng = session_data.get("average_engagement", 0.7)
            if avg_eng < 0.5:
                insight = "Attention drop detected across participants."
                recommendation = "Introduce an interactive question, short break, or change session pacing."
            elif avg_eng > 0.8:
                insight = "High engagement and active participation observed."
                recommendation = "Maintain current instructional pace and discussion flow."
            else:
                insight = "Class status is stable."
                recommendation = "Maintain current pace."

            return {
                "insight": insight,
                "recommendation": recommendation,
                "fallback": True,
                "error": str(e)
            }
