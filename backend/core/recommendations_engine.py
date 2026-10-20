class RecommendationsEngine:
    def generate_realtime_recommendations(self, session_status: dict):
        """
        Rule-based recommendations for immediate feedback.
        Input: Session status dict (people, emotions, attention)
        Output: List of { type: 'alert'|'warning'|'info', message: str }
        """
        recs = []

        # 1. Low Attention Alert
        avg_att = 0
        people = session_status.get('people', [])
        if people:
            avg_att = sum(p['attention'] for p in people) / len(people)

            # Individual check
            for p in people:
                if p['attention'] < 40:
                    recs.append({
                        'type': 'alert',
                        'message': f"Student {p['id']} is distracted ({p['attention']}%)."
                    })

        # 2. Global Attention Warning
        if len(people) > 2 and avg_att < 60:
            recs.append({
                'type': 'warning',
                'message': f"Overall class attention is low ({int(avg_att)}%). Consider an interactive activity or question."
            })

        # 3. Peak Attention Recognition
        if len(people) > 0 and avg_att >= 80:
            recs.append({
                'type': 'info',
                'message': f"Classroom engagement is high ({int(avg_att)}%). Great pacing!"
            })

        return recs
