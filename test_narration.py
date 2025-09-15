from tts_dashboard import narrate_with_elevenlabs

# 🔧 Replace with your actual test values
TEST_TEXT = "This is a test narration from ElevenLabs."
OUTPUT_PATH = "test_output.mp3"
VOICE_ID = "nPczCjzI2devNBz1zQrb"
API_KEY = "sk_a81843c8f22fa2191725c03611f2fb138bf10c73536f4e82"

print("🚀 Starting ElevenLabs narration test...")

success = narrate_with_elevenlabs(TEST_TEXT, OUTPUT_PATH, VOICE_ID, API_KEY)

if success:
    print("✅ Narration test succeeded.")
else:
    print("❌ Narration test failed.")
