import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class AISuggestionEngine:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        try:
            self.client = Groq(api_key=self.api_key) if self.api_key else None
        except Exception:
            self.client = None

    def generate_suggestions(self, session_data: dict):
        """
        Generates actionable teaching suggestions based on fused multimodal metrics.
        """
        prompt = (
            f"Classroom Session Multimodal Data:\n"
            f"- Total Attendance / Participants: {session_data.get('attendance_count', session_data.get('participants_count', 0))}\n"
            f"- Visual Engagement Index: {session_data.get('visual_engagement', session_data.get('average_engagement', 0.0)):.2f}\n"
            f"- Audio Activity & Noise Level: {session_data.get('audio_level', session_data.get('audio_activity', 'moderate'))}\n"
            f"- Speech-to-Silence Ratio: {session_data.get('speech_ratio', 'balanced')}\n"
            f"- Attention Drops / Inactivity: {session_data.get('inactivity_events', 0)}\n\n"
            f"Provide 2-3 specific, actionable teaching suggestions for the instructor to improve engagement and pacing."
        )

        try:
            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert AI teaching assistant."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=200
            )

            response = completion.choices[0].message.content
            return self._parse_response(response)

        except Exception as e:
            print(f"AI Suggestion Error: {e}")
            return self._fallback_suggestions(session_data)

    def _parse_response(self, response: str) -> dict:
        suggestions = [s.strip("- *0123456789.") for s in response.strip().split("\n") if s.strip()]
        return {
            "suggestions": suggestions,
            "raw_response": response,
            "status": "success"
        }

    def _fallback_suggestions(self, session_data: dict) -> dict:
        avg_eng = session_data.get("visual_engagement", session_data.get("average_engagement", 0.7))
        audio_act = session_data.get("audio_level", "moderate")

        suggestions = []
        if avg_eng < 0.5:
            suggestions.append("Engagement is below threshold: ask an open-ended question or conduct a quick poll.")
            suggestions.append("Check for screen/slide fatigue and consider a 1-minute stretch or break.")
        else:
            suggestions.append("Overall learner attention is strong; maintain current instructional pace.")

        if audio_act in ["very_low", "silent"]:
            suggestions.append("Classroom is quiet: prompt a pair-and-share or student discussion.")
        elif audio_act in ["very_high", "noisy"]:
            suggestions.append("High ambient noise detected: confirm students can hear clearly and refocus attention.")

        return {
            "suggestions": suggestions,
            "fallback": True,
            "status": "fallback"
        }
