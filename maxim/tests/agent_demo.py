#!/usr/bin/env python3
"""
Maxim Agent Demo Script

This script demonstrates a multi-step OpenAI agent with comprehensive observability
using Maxim decorators. The agent can handle weather queries, restaurant searches,
and calendar checks using multiple tool calls.

The demo runs through 4 pre-built scenarios showcasing different agent capabilities.

Usage:
    python agent_demo.py

Requirements:
    - OPENAI_API_KEY environment variable (optional - will use mock responses if not provided)
    - MAXIM_API_KEY environment variable
    - MAXIM_BASE_URL environment variable
    - MAXIM_LOG_REPO_ID environment variable
"""

import json
import os
from typing import Dict, List, Any

import openai
from dotenv import load_dotenv

from maxim import Maxim
from maxim.decorators import current_trace, current_generation, generation, span, tool_call, trace

# Load environment variables
load_dotenv()

# Environment setup
openai_api_key = os.getenv("OPENAI_API_KEY")
maxim_api_key = os.getenv("MAXIM_API_KEY")
maxim_base_url = os.getenv("MAXIM_BASE_URL")
maxim_repo_id = os.getenv("MAXIM_LOG_REPO_ID")


class MaximAgent:
    """A multi-tool agent with Maxim observability integration."""

    def __init__(self):
        """Initialize the agent with Maxim logger and OpenAI client."""
        self.maxim = Maxim()
        self.logger = self.maxim.logger()
        
        # Initialize OpenAI client if API key is available
        if openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
            print("✅ OpenAI client initialized - will use real AI responses")
        else:
            self.openai_client = None
            print("⚠️  OpenAI API key not found - will use mock responses")

    @tool_call(
        name="weather_lookup",
        description="Get current weather information for a specific city",
        arguments='{"city": "San Francisco", "units": "celsius"}'
    )
    def get_weather(self, city: str, units: str = "celsius") -> Dict[str, Any]:
        """Get weather information for a city."""
        print(f"🌤️  Looking up weather for {city}...")
        
        # Simulate weather API call
        weather_data = {
            "city": city,
            "temperature": 22 if units == "celsius" else 72,
            "units": units,
            "condition": "partly cloudy",
            "humidity": 65,
            "wind_speed": 12,
            "timestamp": "2024-01-20T14:30:00Z"
        }
        
        print(f"   Weather: {weather_data['temperature']}°{weather_data['units'][0].upper()}, {weather_data['condition']}")
        return weather_data

    @tool_call(
        name="restaurant_search",
        description="Search for restaurants in a specific area with cuisine preferences",
        arguments='{"location": "downtown", "cuisine": "italian", "price_range": "moderate"}'
    )
    def search_restaurants(self, location: str, cuisine: str = "any", price_range: str = "moderate") -> List[Dict[str, Any]]:
        """Search for restaurants based on criteria."""
        print(f"🍽️  Searching for {cuisine} restaurants in {location}...")
        
        # Simulate restaurant search API
        restaurants = [
            {
                "name": f"Bella {cuisine.title()} Bistro",
                "location": location,
                "cuisine": cuisine,
                "price_range": price_range,
                "rating": 4.5,
                "address": f"123 Main St, {location}",
                "phone": "+1-555-0123"
            },
            {
                "name": f"The {cuisine.title()} Corner",
                "location": location,
                "cuisine": cuisine,
                "price_range": price_range,
                "rating": 4.2,
                "address": f"456 Oak Ave, {location}",
                "phone": "+1-555-0456"
            }
        ]
        
        print(f"   Found {len(restaurants)} restaurants")
        return restaurants

    @tool_call(
        name="calendar_check",
        description="Check calendar availability for scheduling",
        arguments='{"date": "2024-01-21", "time_slot": "evening"}'
    )
    def check_calendar(self, date: str, time_slot: str = "any") -> Dict[str, Any]:
        """Check calendar availability."""
        print(f"📅 Checking calendar availability for {date} ({time_slot})...")
        
        availability = {
            "date": date,
            "time_slot": time_slot,
            "available": True,
            "suggested_times": ["6:00 PM", "7:00 PM", "8:00 PM"],
            "conflicts": [],
            "timezone": "PST"
        }
        
        print(f"   Available: {availability['available']}")
        return availability

    @span(name="information_gathering")
    def gather_information(self, user_request: str) -> Dict[str, Any]:
        """Gather information using multiple tool calls based on user request."""
        print(f"🔍 Analyzing request: '{user_request}'")
        gathered_info = {}
        
        # Analyze request to determine which tools to use
        request_lower = user_request.lower()
        
        if "weather" in request_lower:
            # Extract city from request (simplified)
            city = "San Francisco"  # Default city
            if "new york" in request_lower:
                city = "New York"
            elif "london" in request_lower:
                city = "London"
            elif "chicago" in request_lower:
                city = "Chicago"
            
            gathered_info["weather"] = self.get_weather(city, "celsius")
        
        if "restaurant" in request_lower or "food" in request_lower or "eat" in request_lower:
            # Extract location and cuisine preferences
            location = "downtown"
            cuisine = "italian"
            
            if "chinese" in request_lower:
                cuisine = "chinese"
            elif "mexican" in request_lower:
                cuisine = "mexican"
            elif "japanese" in request_lower:
                cuisine = "japanese"
            elif "thai" in request_lower:
                cuisine = "thai"
            
            gathered_info["restaurants"] = self.search_restaurants(location, cuisine)
        
        if "schedule" in request_lower or "calendar" in request_lower or "available" in request_lower:
            gathered_info["calendar"] = self.check_calendar("2024-01-21", "evening")
        
        print(f"📊 Gathered information from {len(gathered_info)} tools")
        return gathered_info

    @generation(name="response_generation")
    def generate_response(self, user_request: str, gathered_info: Dict[str, Any]) -> str:
        """Generate response using OpenAI based on gathered information."""
        print("🤖 Generating AI response...")
        
        if not self.openai_client:
            # Mock response when OpenAI is not available
            mock_input = {
                "user_request": user_request,
                "gathered_info": gathered_info,
                "provider": "mock"
            }
            
            # Track mock generation input
            current_generation().set_provider("mock")
            current_generation().set_model("mock-response-generator")
            
            response_parts = []
            
            if "weather" in gathered_info:
                weather = gathered_info["weather"]
                response_parts.append(
                    f"The weather in {weather['city']} is {weather['temperature']}°{weather['units'][0].upper()} "
                    f"with {weather['condition']} conditions."
                )
            
            if "restaurants" in gathered_info:
                restaurants = gathered_info["restaurants"]
                response_parts.append(
                    f"I found {len(restaurants)} restaurants for you. "
                    f"Top recommendation: {restaurants[0]['name']} with a {restaurants[0]['rating']} rating."
                )
            
            if "calendar" in gathered_info:
                calendar = gathered_info["calendar"]
                if calendar["available"]:
                    response_parts.append(
                        f"You're available on {calendar['date']} during {calendar['time_slot']}. "
                        f"Suggested times: {', '.join(calendar['suggested_times'])}."
                    )
            
            response = " ".join(response_parts) if response_parts else "I've processed your request."
            
            # Track mock generation result
            current_generation().result(response)
            
            print("   Using mock response (OpenAI not available)")
            return response
        
        # Real OpenAI call
        try:
            system_prompt = """You are a helpful assistant that provides personalized recommendations 
            based on gathered information. Be concise but informative in your responses."""
            
            user_prompt = f"""
            User request: {user_request}
            
            Gathered information: {json.dumps(gathered_info, indent=2)}
            
            Please provide a helpful response based on this information.
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Track generation input
            current_generation().set_provider("openai")
            current_generation().set_model("gpt-3.5-turbo")
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=300,
                temperature=0.7
            )
      
            current_generation().result(response)
            
            ai_response = response.choices[0].message.content or "I apologize, but I couldn't generate a response."
            print("   Using OpenAI generated response")
            return ai_response
            
        except (openai.OpenAIError, Exception) as e:
            error_response = f"Error generating response: {str(e)}"
            
            # Track error generation
            error_input = {
                "user_request": user_request,
                "gathered_info": gathered_info,
                "error": str(e)
            }
            current_generation().set_provider("openai")
            current_generation().set_model("gpt-3.5-turbo")
            current_generation().result(error_response)
            current_generation().add_tag("error", "true")
            
            print(f"   Error: {error_response}")
            return error_response

    @span(name="request_processing")
    def process_user_request(self, user_request: str) -> Dict[str, Any]:
        """Process user request through multiple steps."""
        print("⚙️  Processing request...")
        
        # Step 1: Gather information
        gathered_info = self.gather_information(user_request)
        
        # Step 2: Generate response
        response = self.generate_response(user_request, gathered_info)
        
        # Step 3: Compile final result
        result = {
            "user_request": user_request,
            "gathered_info": gathered_info,
            "response": response,
            "processing_steps": ["information_gathering", "response_generation", "result_compilation"]
        }
        
        return result

    def run_conversation(self, user_request: str, session_id: str = "demo_session") -> Dict[str, Any]:
        """Main agent conversation flow with full observability."""
        
        @trace(name="agent_conversation", logger=self.logger)
        def _traced_conversation():
            print(f"\n🚀 Starting agent conversation (Session: {session_id})")
            print(f"📝 User Request: {user_request}")
            print("-" * 60)
            
            current_trace_obj = current_trace()
            if current_trace_obj:
                current_trace_obj.set_input({"user_request": user_request, "session_id": session_id})
                current_trace_obj.add_tag("agent_type", "multi_tool_assistant")
                current_trace_obj.add_tag("session_id", session_id)
            
            try:
                # Process the user request
                result = self.process_user_request(user_request)
                
                if current_trace_obj:
                    current_trace_obj.set_output(result)
                
                print("-" * 60)
                print("✅ Conversation completed successfully!")
                return result
                
            except (ValueError, RuntimeError, Exception) as e:
                error_result = {
                    "error": str(e),
                    "user_request": user_request,
                    "status": "failed"
                }
                
                if current_trace_obj:
                    current_trace_obj.set_output(error_result)
                    current_trace_obj.add_tag("error", "true")
                
                print(f"❌ Error occurred: {str(e)}")
                return error_result
        
        return _traced_conversation()

    def flush_logs(self):
        """Flush logs to Maxim."""
        print("📤 Sending logs to Maxim...")
        self.logger.flush()
        print("✅ Logs sent successfully!")

    def cleanup(self):
        """Clean up resources."""
        self.flush_logs()
        self.maxim.cleanup()


def print_banner():
    """Print a welcome banner."""
    print("=" * 70)
    print("🤖 MAXIM AGENT DEMO")
    print("=" * 70)
    print("This demo showcases a multi-tool AI agent with full observability")
    print("using Maxim decorators for tracing, spans, and tool calls.")
    print()
    print("Running 4 demo scenarios to showcase agent capabilities...")
    print()


def print_result(result: Dict[str, Any]):
    """Print the conversation result in a formatted way."""
    print("\n📋 CONVERSATION RESULT:")
    print("-" * 40)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print("🎯 Final Response:")
    print(f"   {result['response']}")
    print()
    
    if "gathered_info" in result:
        print("📊 Information Gathered:")
        for tool_name, data in result["gathered_info"].items():
            print(f"   • {tool_name.title()}: {type(data).__name__} with {len(data) if isinstance(data, (list, dict)) else 1} item(s)")
    
    print()


def run_demo_scenarios(agent: MaximAgent):
    """Run several demo scenarios to showcase agent capabilities."""
    scenarios = [
        {
            "name": "Weather Query",
            "request": "What's the weather like in New York today?",
            "session_id": "weather_demo"
        },
        {
            "name": "Restaurant Search",
            "request": "I'm looking for good Chinese restaurants downtown for dinner",
            "session_id": "restaurant_demo"
        },
        {
            "name": "Multi-Tool Request",
            "request": "What's the weather in Chicago and can you find Italian restaurants? Also check if I'm available tomorrow evening.",
            "session_id": "multi_tool_demo"
        },
        {
            "name": "Calendar Check",
            "request": "Am I available for a meeting tomorrow evening?",
            "session_id": "calendar_demo"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🎬 DEMO SCENARIO {i}: {scenario['name']}")
        print("=" * 50)
        
        result = agent.run_conversation(scenario["request"], scenario["session_id"])
        print_result(result)
        
        if i < len(scenarios):
            input("\nPress Enter to continue to the next scenario...")


def main():
    """Main function to run the agent demo."""
    print_banner()
    
    # Check environment variables
    if not maxim_api_key:
        print("⚠️  Warning: MAXIM_API_KEY not found. Logs may not be sent to Maxim.")
    
    # Initialize agent
    print("🔧 Initializing Maxim Agent...")
    agent = MaximAgent()
    print("✅ Agent initialized successfully!")
    
    try:
        # Run demo scenarios
        run_demo_scenarios(agent)
    
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        agent.cleanup()
        print("✅ Demo completed!")


if __name__ == "__main__":
    main()
