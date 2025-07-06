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
        # 将numpy数组转换回PIL图像
        image = Image.fromarray(image[:, :, :3], mode='RGB')

        # 将图像保存到字节流中
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")

        # 将字节流编码为Base64
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"Failed to convert image to base64: {e}")

class SiliconFlowVLM:
    """
    SiliconFlow API implementation for VLM using free models
    """

    def __init__(self, model="Qwen/Qwen2.5-VL-7B-Instruct", system_instruction=None):
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