import json
import re
import logging

logger = logging.getLogger(__name__)


def parse_response(response_content: str):
    try:
        # Attempt to parse the JSON response
        response_data = json.loads(response_content)
        return (
            response_data,
            True,
        )  # Return the parsed data and a flag indicating successful parsing
    except json.JSONDecodeError:
        # Fallback to regex if JSON parsing fails
        try:
            message_for_user = re.search(
                r'"message_for_user":\s*"([^"]*)"', response_content
            ).group(1)
            message_for_system = re.search(
                r'"message_for_system":\s*"([^"]*)"', response_content
            ).group(1)
            return {
                "message_for_user": message_for_user,
                "message_for_system": message_for_system,
            }, False
        except Exception as e:
            logger.error(f"Failed to extract using fallback method: {e}")
            return {
                "message_for_user": "",
                "message_for_system": "",
            }, False  # Return empty values as a last resort


# Example usage
ai_message_content = """
{
  "message_for_user": "Great! Here is the draft synopsis for our story:

'Aleksandr Jojo and the Quest for the Midnight Sun'

In a quiet suburb where the real and fantastical blur, a seven-year-old boy named Aleksandr Jojo finds wonder in the ordinary. With his tall stature, blond hair, and intelligent brown eyes, he constructs incredible structures with his Legos and explores vast worlds in Minecraft. His best friend, Carlos, shares in these creative adventures, but it’s on one unassuming afternoon that their lives take an extraordinary turn.
 
While playing in Aleksandr's room, the pair uncovers a mystical gateway hidden within a Lego castle, leading them to a world that mirrors the fantastical dimensions of Minecraft and Star Wars. Here, they meet the hilariously eccentric Pirate Blackberry Hook, who speaks only in delightful Spanish quips, and a magical pink dragon that breathes not fire, but magical dust capable of constructing wonders.

As Aleksandr and Carlos traverse this new world, they encounter various challenges that require them to combine their ingenuity, courage, and friendship. Blackberry Hook, with his uproarious banter, aids them in their quest to find the 'Midnight Sun,' a fabled source of light that can banish darkness—not only in their fantastical journey but within Aleksandr's heart, addressing his fear of the dark.

Their journey is punctuated by raucous dialogue and jest, as they decipher riddles, construct marvels, and ultimately face the very embodiment of Aleksandr's fear—a dark entity threatening to extinguish the light of creativity and joy in their worlds. In an epic clash of wit and will, the friends, alongside their peculiar allies, confront the antagonist.

The climax of the story culminates in a clever stratagem, devised by the friends and executed with the help of Blackberry Hook's humor and the dragon's magical dust. Together, they illuminate the darkness, revealing that bravery and laughter can overcome any fear.

As the adventure concludes, Aleksandr gains a newfound confidence that light and friendship can dispel any shadows, emerging as a hero not only in this story but in his personal world. Their success demonstrates the power of creativity, unity across cultures, and the strength found in embracing one's unique traits.

— End of Synopsis —

Let me know if this captures the spirit of what you're envisioning, or if there are any adjustments you would like to integrate!",
  "message_for_system": ""
}
"""

parsed_response, was_json = parse_response(ai_message_content)
if not was_json:
    logger.warning("Had to use fallback method for parsing.")

print(f"Message for human: {parsed_response['message_for_user']}")
print(f"Message for system: {parsed_response['message_for_system']}")
