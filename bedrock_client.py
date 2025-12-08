import boto3
import json

class BedrockClient:
    def __init__(self):
        # DOUBLE CHECK: Are you in us-east-1 (N. Virginia)? 
        # If your AWS console says Oregon, change this to 'us-west-2'
        self.client = boto3.client(
            service_name='bedrock-runtime', 
            region_name='us-east-1' 
        )

    def get_embedding(self, text):
        """
        Uses Amazon Titan Embeddings G1 - Text
        """
        body = json.dumps({"inputText": text})
        try:
            response = self.client.invoke_model(
                modelId='amazon.titan-embed-text-v1',
                contentType='application/json',
                accept='application/json',
                body=body
            )
            response_body = json.loads(response['body'].read())
            return response_body['embedding']
        except Exception as e:
            print(f"⚠️ Embedding Error: {e}")
            return [0.0] * 1536 

    def generate_text(self, prompt):
        """
        Uses Amazon Titan Text Express (The Backup Plan)
        """
        # Titan uses a different JSON structure than Claude
        body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 512,
                "temperature": 0.0,  # 0.0 makes it more deterministic (better for JSON)
                "topP": 1
            }
        })

        try:
            response = self.client.invoke_model(
                modelId='amazon.titan-text-express-v1',
                contentType='application/json',
                accept='application/json',
                body=body
            )
            response_body = json.loads(response['body'].read())
            return response_body['results'][0]['outputText']
        except Exception as e:
            print(f"⚠️ Generation Error (Titan): {e}")
            return "Error generating text."

    def validate_review(self, text, rating):
        """
        The Semantic Judge (Titan Version).
        """
        # Titan needs very clear instructions to output strict JSON
        prompt = f"""
        You are a Data Quality Bot. Analyze this data.
        
        DATA:
        Review: "{text}"
        Rating: {rating}
        
        RULES:
        1. PII: Phone numbers or emails?
        2. Mismatch: Negative text but 5 stars?
        3. Bad Data: Gibberish?

        OUTPUT format must be strictly JSON:
        {{
            "is_valid": true or false,
            "reason": "Clean" or "PII Detected" or "Sentiment Mismatch"
        }}
        
        JSON:
        """
        
        try:
            # 1. Get raw text
            raw_response = self.generate_text(prompt)
            print(f"🔍 Titan Judge Output: {raw_response}") 
            
            # 2. Parse JSON (Titan often adds text, so we clean it)
            json_str = raw_response[raw_response.find('{'):raw_response.rfind('}')+1]
            return json.loads(json_str)
            
        except Exception as e:
            print(f"❌ Titan Parsing Failed: {e}")
            # If Titan fails to write JSON, we treat it as a Block just to be safe/show in UI
            return {
                "is_valid": False, 
                "reason": "AI_Parsing_Error"
            }