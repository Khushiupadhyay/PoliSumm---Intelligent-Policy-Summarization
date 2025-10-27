"""Quick script to show the summary"""
import json

with open('summary.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("="*60)
print("POLISUMM SUMMARY")
print("="*60)
print()
print(data['summary'])
print()
print("Statistics:")
print(f"  Total sentences: {data['statistics']['total_sentences']}")
print(f"  Total words: {data['statistics']['total_words']}")
print(f"  Preservation score: {data['evaluation'].get('preservation_score', 0):.3f}")

