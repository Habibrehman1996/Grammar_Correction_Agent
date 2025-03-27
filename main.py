import os
import chainlit as cl
import google.generativeai as genai

from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

gemini_api_key = "AIzaSyDkN0ptEcPh7pfoyX7Re0zmookEO5cnV_0"



provider =  AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
    
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider,
)

run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

agent = Agent(
             name="Habib Assistant Agent", 
             instructions="You are a helpful assistant that can correct user grammatical mistakes.Sends user input to Gemini API and returns the corrected grammar.", 
)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello, I'm the F&F Developers Grammatical Correction Agent. How may I help you today?").send()
def get_gemini_response(prompt):
    """Sends user input to Gemini API and returns the corrected grammar."""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

@cl.on_message
async def handle_message(message: cl.Message):
    history= cl.user_session.get("history")

    user_input = message.content
    corrected_text = get_gemini_response(f"Correct the grammar and provide feedback: {user_input}")
    await cl.Message(content=f"📝 **Correction:** {corrected_text}").send()

    #msg.streamed_chunk
    
    history.append({"role":"user", "content": message.content})
    result = Runner.run_streamed(
        agent,
        input=history, 
        run_config=run_config,
    
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
           # print(event.data.delta, end="", flush=True)
            #msg.update(event.data.delta)
            await msg.stream_token(event.data.delta)
   
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:cl.app", host="0.0.0.0", port=8000, reload=False)
    #await cl.Message(content=result.final_output).send()
