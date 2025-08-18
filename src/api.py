import logging
from openai import OpenAI
import base64
import numpy as np
from PIL import Image
import io
import os
import httpx
import time

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

def encode_image(image):
    try:
        # Handle different input types
        if isinstance(image, (list, tuple)):
            if len(image) > 0:
                image = image[0]  # Take first image if it's a list
            else:
                raise ValueError("Empty image list provided")
                
        # Convert numpy array if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
            
        # Ensure we have a proper RGB image
        if len(image.shape) == 3 and image.shape[2] >= 3:
            image = image[:, :, :3]  # Take only RGB channels
        
        # Convert to PIL Image
        image = Image.fromarray(image.astype(np.uint8))
        
        # Save to bytes buffer
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        
        # Return base64 encoded string
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        error_msg = f"Failed to convert image to base64: {str(e)}"
        print(f"❌ {error_msg}")
        logging.error(error_msg)
        # Return a placeholder image to avoid breaking the flow
        return None

class SiliconFlowVLM:
    """
    SiliconFlow API implementation for VLM using free models
    """

    def __init__(self, model="Pro/Qwen/Qwen2.5-VL-7B-Instruct", system_instruction=None):
        """
        Initialize the SiliconFlow model with specified configuration.

        Parameters
        ----------
        model : str
            The model version to be used. For VLM: "Qwen/Qwen2.5-VL-7B-Instruct"
            For text-only testing: "Qwen/Qwen2.5-7B-Instruct" (free)
        system_instruction : str, optional
            System instructions for model behavior.
        """
        self.name = model
        
        # Store original proxy environment variables
        original_proxies = {}
        proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']
        
        # Save and clear proxy environment variables for API calls
        for var in proxy_vars:
            if var in os.environ:
                original_proxies[var] = os.environ[var]
                del os.environ[var]
        
        try:
            # Create client with SiliconFlow API
            self.client = OpenAI(
                api_key=os.environ.get("SILICONFLOW_API_KEY"),  # 需要设置这个环境变量
                base_url="https://api.siliconflow.cn/v1"
            )
        finally:
            # Restore original proxy settings
            for var, value in original_proxies.items():
                os.environ[var] = value

        self.system_instruction = system_instruction
        self.spend = 0  # SiliconFlow免费模型暂时不计费

    def _create_messages(self, text_prompt: str, base64_image: str = None):
        """Create messages array for API call"""
        messages = []
        
        if self.system_instruction:
            messages.append({
                "role": "system", 
                "content": self.system_instruction
            })
        
        # Check if model supports vision
        if "VL" in self.name and base64_image:
            # Vision model with image
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    },
                ],
            })
        else:
            # Text-only model or no image provided
            messages.append({
                "role": "user",
                "content": text_prompt
            })
        
        return messages

    def call_chat(self, image: list[np.array], text_prompt: str):
        """Chat completion with system instruction"""
        base64_image = None
        if image and len(image) > 0:
            try:
                base64_image = encode_image(image[0])
            except Exception as e:
                print(f"Image encoding error: {e}")
        
        messages = self._create_messages(text_prompt, base64_image)
        
        # 临时清除代理设置
        original_proxies = {}
        proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']
        
        for var in proxy_vars:
            if var in os.environ:
                original_proxies[var] = os.environ[var]
                del os.environ[var]
        
        try:
            response = self.client.chat.completions.create(
                model=self.name,
                messages=messages,
                max_tokens=500,
                temperature=0.1,  # 降低随机性以获得更稳定的结果
                top_p=0.9,
                stream=False,
                timeout=30  # 添加超时设置
            )
            
            # 对于免费模型，我们简单计算token数量
            if hasattr(response, 'usage') and response.usage:
                self.spend += (response.usage.prompt_tokens + response.usage.completion_tokens)
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"SILICONFLOW API ERROR: {e}")
            return "SILICONFLOW API ERROR"
        finally:
            # 恢复代理设置
            for var, value in original_proxies.items():
                os.environ[var] = value

    def call(self, image: list[np.array], text_prompt: str):
        """Basic completion without system instruction"""
        base64_image = None
        if image and len(image) > 0:
            try:
                base64_image = encode_image(image[0])
            except Exception as e:
                print(f"Image encoding error: {e}")
        
        messages = self._create_messages(text_prompt, base64_image)
        
        # 临时清除代理设置
        original_proxies = {}
        proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']
        
        for var in proxy_vars:
            if var in os.environ:
                original_proxies[var] = os.environ[var]
                del os.environ[var]
        
        try:
            # Add retry mechanism
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.name,
                        messages=messages,
                        max_tokens=500,
                        temperature=0.1,
                        top_p=0.9,
                        stream=False,
                        timeout=30
                    )
                    
                    if hasattr(response, 'usage') and response.usage:
                        self.spend += (response.usage.prompt_tokens + response.usage.completion_tokens)
                    
                    return response.choices[0].message.content
                    
                except Exception as e:
                    print(f"SILICONFLOW API ERROR (Attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
            
            # If all retries fail, return a parseable fallback response
            print("All API attempts failed, returning fallback response")
            return '{"action": 1, "reason": "API connection failed, using default forward action"}'
        finally:
            # 恢复代理设置
            for var, value in original_proxies.items():
                os.environ[var] = value

    def reset(self):
        """Reset the context state of the VLM agent."""
        pass

    def get_spend(self):
        """Retrieve the total token usage."""
        return self.spend

class SiliconFlowLLM:
    """
    SiliconFlow API implementation for pure LLM text generation
    Used for goal subgraph construction and semantic reasoning
    """

    def __init__(self, model="Pro/Qwen/Qwen2.5-7B-Instruct", system_instruction=None):
        """
        Initialize the SiliconFlow LLM with specified configuration.

        Parameters
        ----------
        model : str
            The LLM model version to be used. Default: "Pro/Qwen/Qwen2.5-7B-Instruct" (free)
        system_instruction : str, optional
            System instructions for model behavior.
        """
        self.name = model
        
        # Store original proxy environment variables
        original_proxies = {}
        proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']
        
        # Save and clear proxy environment variables for API calls
        for var in proxy_vars:
            if var in os.environ:
                original_proxies[var] = os.environ[var]
                del os.environ[var]
        
        try:
            # Initialize OpenAI client with SiliconFlow endpoint
            self.client = OpenAI(
                api_key=os.getenv("SILICONFLOW_API_KEY", "sk-placeholder"), 
                base_url="https://api.siliconflow.cn/v1"
            )
        finally:
            # Restore proxy environment variables
            for var, value in original_proxies.items():
                os.environ[var] = value

        self.system_instruction = system_instruction
        self.spend = 0  # SiliconFlow免费模型暂时不计费

    def _create_messages(self, text_prompt: str):
        """Create messages array for LLM API call"""
        messages = []
        
        if self.system_instruction:
            messages.append({
                "role": "system",
                "content": self.system_instruction
            })
        
        messages.append({
            "role": "user", 
            "content": text_prompt
        })
        
        return messages

    def call_chat(self, text_prompt: str):
        """Chat completion with system instruction"""
        messages = self._create_messages(text_prompt)
        
        # 临时清除代理设置
        original_proxies = {}
        proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']
        
        for var in proxy_vars:
            if var in os.environ:
                original_proxies[var] = os.environ[var]
                del os.environ[var]
        
        if original_proxies:
            print(f"🌐 SiliconFlow LLM: Temporarily cleared proxy settings: {list(original_proxies.keys())}")
        
        try:
            response = self.client.chat.completions.create(
                model=self.name,
                messages=messages,
                temperature=0.7,
                max_tokens=2048
            )
            print(f"✅ SiliconFlow LLM API call successful")
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ SiliconFlow LLM API error: {e}")
            logging.error(f"SiliconFlow LLM API error: {e}")
            return f"API调用失败: {str(e)}"
        finally:
            # 恢复代理设置
            for var, value in original_proxies.items():
                os.environ[var] = value
            if original_proxies:
                print(f"🔄 SiliconFlow LLM: Restored proxy settings: {list(original_proxies.keys())}")

    def call(self, text_prompt: str):
        """Basic completion without system instruction"""
        return self.call_chat(text_prompt)

    def reset(self):
        """Reset the context state of the LLM agent."""
        pass

    def get_spend(self):
        """Retrieve the total token usage."""
        return self.spend

# 保持原有的类作为备选
class QwenVLM:
    """原有的Qwen实现作为备选"""

    def __init__(self, model="qwen-vl-plus", system_instruction=None):
        """
        Initialize the Qwen model with specified configuration.

        Parameters
        ----------
        model : str
            The model version to be used.
        system_instruction : str, optional
            System instructions for model behavior.
        """
        self.name = model
        
        # Store original proxy environment variables
        original_proxies = {}
        proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']
        
        # Save and clear proxy environment variables
        for var in proxy_vars:
            if var in os.environ:
                original_proxies[var] = os.environ[var]
                del os.environ[var]
        
        try:
            # Create client with proxy environment variables cleared
            self.client = OpenAI(
                api_key=os.environ.get("GEMINI_API_KEY"),
                base_url=os.environ.get("GEMINI_BASE_URL")
            )
        finally:
            # Restore original proxy settings
            for var, value in original_proxies.items():
                os.environ[var] = value

        self.system_instruction = system_instruction
        self.spend = 0

    def call_chat(self, image: list[np.array], text_prompt: str):
        base64_image = encode_image(image[0])
        try:
            response = self.client.chat.completions.create(
                model=self.name,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_instruction
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/png;base64,{base64_image}"
                            },
                        ],
                    }
                ],
                max_tokens=500,
                temperature=0,
                top_p=1,
                stream=False  # 是否开启流式输出
            )
            self.spend += (response.usage.prompt_tokens + response.usage.completion_tokens)

        except Exception as e:
            print(f"GEMINI API ERROR: {e}")
            return "GEMINI API ERROR"
        return response.choices[0].message.content

    def call(self, image: list[np.array], text_prompt: str):
        base64_image = encode_image(image[0])
        try:
            response = self.client.chat.completions.create(
                model=self.name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                            {
                                "type": "image_url",
                                # Fix the image_url format:
                                "image_url": f"data:image/png;base64,{base64_image}"
                            },
                        ],
                    }
                ],
                max_tokens=500,
                temperature=0,
                top_p=1,
                stream=False  # 是否开启流式输出
            )
            self.spend += (response.usage.prompt_tokens + response.usage.completion_tokens)
        except Exception as e:
            print(f"GEMINI API ERROR: {e}")
            return "GEMINI API ERROR"
        return response.choices[0].message.content

    def reset(self):
        """
        Reset the context state of the VLM agent.
        """
        pass

    def get_spend(self):
        """
        Retrieve the total spend on model usage.
        """
        return self.spend

class GeminiVLM:
    """
    A specific implementation of a VLM using Google's official Gemini API for image and text inference.
    Requires proxy for accessing Google services in China.
    """

    def __init__(self, model="gemini-1.5-pro", system_instruction=None):
        """
        Initialize the Gemini model with specified configuration.

        Parameters
        ----------
        model : str
            The model version to be used. Default: gemini-1.5-pro
        system_instruction : str, optional
            System instructions for model behavior.
        """
        self.name = model
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.base_url = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
        self.system_instruction = system_instruction
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        self.spend = 0
        # Pricing for Gemini models (per million tokens)
        if '1.5-pro' in self.name:
            self.cost_per_input_token = 1.25 / 1_000_000
            self.cost_per_output_token = 5.0 / 1_000_000
        elif '1.5-flash' in self.name:
            self.cost_per_input_token = 0.075 / 1_000_000
            self.cost_per_output_token = 0.3 / 1_000_000
        else:
            self.cost_per_input_token = 0.5 / 1_000_000
            self.cost_per_output_token = 1.5 / 1_000_000

    def _setup_proxy_for_google(self):
        """Setup proxy for Google API access"""
        return {
            'http': 'http://127.0.0.1:7897',
            'https': 'http://127.0.0.1:7897'
        }

    def _create_google_payload(self, text_prompt: str, base64_image: str = None):
        """Create payload for Google's Gemini API"""
        parts = []
        
        # Add system instruction if provided
        if self.system_instruction:
            parts.append({"text": f"System: {self.system_instruction}\n\nUser: {text_prompt}"})
        else:
            parts.append({"text": text_prompt})
        
        # Add image if provided
        if base64_image:
            parts.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": base64_image
                }
            })
        
        return {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "maxOutputTokens": 500,
                "temperature": 0.1,
                "topP": 0.9
            }
        }

    def _make_request(self, payload):
        """Make request to Google's Gemini API with proxy"""
        import requests
        import json
        
        url = f"{self.base_url}/models/{self.name}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        
        # Use proxy for Google API
        proxies = self._setup_proxy_for_google()
        
        try:
            print(f"🌐 Making Gemini API request with proxy: {proxies}")
            response = requests.post(url, headers=headers, json=payload, proxies=proxies, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            print(f"📊 Gemini API response status: {response.status_code}")
            
            # Extract text from response
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    text_parts = [part.get('text', '') for part in candidate['content']['parts'] if 'text' in part]
                    response_text = ''.join(text_parts)
                    
                    # Update spending (approximate token count)
                    if 'usageMetadata' in result:
                        usage = result['usageMetadata']
                        input_tokens = usage.get('promptTokenCount', 0)
                        output_tokens = usage.get('candidatesTokenCount', 0)
                        self.spend += (input_tokens * self.cost_per_input_token + 
                                     output_tokens * self.cost_per_output_token)
                        print(f"💰 Token usage: input={input_tokens}, output={output_tokens}, cost=${self.spend:.4f}")
                    
                    return response_text
            
            print("⚠️ No valid response content found")
            return "No valid response from Gemini API"
            
        except requests.exceptions.RequestException as e:
            error_msg = f"GEMINI API ERROR: {e}"
            print(f"❌ {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"GEMINI API PARSING ERROR: {e}"
            print(f"❌ {error_msg}")
            return error_msg

    def call_chat(self, image: list[np.array], text_prompt: str):
        base64_image = None
        if image and len(image) > 0:
            base64_image = encode_image(image[0])
        
        payload = self._create_google_payload(text_prompt, base64_image)
        return self._make_request(payload)

    def call(self, image: list[np.array], text_prompt: str):
        base64_image = None
        if image and len(image) > 0:
            base64_image = encode_image(image[0])
        
        payload = self._create_google_payload(text_prompt, base64_image)
        return self._make_request(payload)

    def reset(self):
        """Reset the context state of the VLM agent."""
        pass

    def get_spend(self):
        """Retrieve the total spend on model usage."""
        return self.spend