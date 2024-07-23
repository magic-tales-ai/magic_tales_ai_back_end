import json
import uuid
import traceback

from openai import OpenAI
from typing_extensions import override
from openai import AssistantEventHandler, OpenAI
from openai.types.beta.threads import Text, TextDelta
from openai.types.beta.threads.runs import ToolCall, ToolCallDelta
from openai.types.beta.threads import Message, MessageDelta
from openai.types.beta.threads.runs import ToolCall, RunStep
from openai.types.beta import AssistantStreamEvent

from services.utils.log_utils import get_logger

# Get a logger instance for this module
logger = get_logger(__name__)


class EventHandler(AssistantEventHandler):
    def __init__(self, thread_id, assistant_id):
        super().__init__()
        self.output = None
        self.tool_id = None
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self.run_id = None
        self.run_step = None
        self.function_name = ""
        self.arguments = ""

    @override
    def on_text_created(self, text) -> None:
        logger.info(f"\nassistant on_text_created > ", end="", flush=True)

    @override
    def on_text_delta(self, delta, snapshot):
        # logger.info(f"\nassistant on_text_delta > {delta.value}", end="", flush=True)
        logger.info(f"{delta.value}")

    @override
    def on_end(
        self,
    ):
        logger.info(
            f"\n end assistant > ", self.current_run_step_snapshot, end="", flush=True
        )

    @override
    def on_exception(self, exception: Exception) -> None:
        """Fired whenever an exception happens during streaming"""
        logger.info(f"\nassistant > {exception}\n", end="", flush=True)

    @override
    def on_message_created(self, message: Message) -> None:
        logger.info(f"\nassistant on_message_created > {message}\n", end="", flush=True)

    @override
    def on_message_done(self, message: Message) -> None:
        logger.info(f"\nassistant on_message_done > {message}\n", end="", flush=True)

    @override
    def on_message_delta(self, delta: MessageDelta, snapshot: Message) -> None:
        # logger.info(f"\nassistant on_message_delta > {delta}\n", end="", flush=True)
        pass

    def on_tool_call_created(self, tool_call):
        # 4
        logger.info(f"\nassistant on_tool_call_created > {tool_call}")
        self.function_name = tool_call.function.name
        self.tool_id = tool_call.id
        logger.info(f"\on_tool_call_created > run_step.status > {self.run_step.status}")

        logger.info(f"\nassistant > {tool_call.type} {self.function_name}\n", flush=True)

        keep_retrieving_run = openai_client.beta.threads.runs.retrieve(
            thread_id=self.thread_id, run_id=self.run_id
        )

        while keep_retrieving_run.status in ["queued", "in_progress"]:
            keep_retrieving_run = openai_client.beta.threads.runs.retrieve(
                thread_id=self.thread_id, run_id=self.run_id
            )

            logger.info(f"\nSTATUS: {keep_retrieving_run.status}")

    @override
    def on_tool_call_done(self, tool_call: ToolCall) -> None:
        keep_retrieving_run = openai_client.beta.threads.runs.retrieve(
            thread_id=self.thread_id, run_id=self.run_id
        )

        logger.info(f"\nDONE STATUS: {keep_retrieving_run.status}")

        if keep_retrieving_run.status == "completed":
            all_messages = openai_client.beta.threads.messages.list(
                thread_id=current_thread.id
            )

            logger.info(all_messages.data[0].content[0].text.value, "", "")
            return

        elif keep_retrieving_run.status == "requires_action":
            logger.info("here you would call your function")

            if self.function_name == "example_blog_post_function":
                function_data = my_example_funtion()

                self.output = function_data

                with openai_client.beta.threads.runs.submit_tool_outputs_stream(
                    thread_id=self.thread_id,
                    run_id=self.run_id,
                    tool_outputs=[
                        {
                            "tool_call_id": self.tool_id,
                            "output": self.output,
                        }
                    ],
                    event_handler=EventHandler(self.thread_id, self.assistant_id),
                ) as stream:
                    stream.until_done()
            else:
                logger.info("unknown function")
                return

    @override
    def on_run_step_created(self, run_step: RunStep) -> None:
        # 2
        logger.info(f"on_run_step_created")
        self.run_id = run_step.run_id
        self.run_step = run_step
        logger.info("The type ofrun_step run step is ", type(run_step), flush=True)
        logger.info(f"\n run step created assistant > {run_step}\n", flush=True)

    @override
    def on_run_step_done(self, run_step: RunStep) -> None:
        logger.info(f"\n run step done assistant > {run_step}\n", flush=True)

    def on_tool_call_delta(self, delta, snapshot):
        if delta.type == "function":
            # the arguments stream thorugh here and then you get the requires action event
            logger.info(delta.function.arguments, end="", flush=True)
            self.arguments += delta.function.arguments
        elif delta.type == "code_interpreter":
            logger.info(f"on_tool_call_delta > code_interpreter")
            if delta.code_interpreter.input:
                logger.info(delta.code_interpreter.input, end="", flush=True)
            if delta.code_interpreter.outputs:
                logger.info(f"\n\noutput >", flush=True)
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        logger.info(f"\n{output.logs}", flush=True)
        else:
            logger.info("ELSE")
            logger.info(delta, end="", flush=True)

    @override
    def on_event(self, event: AssistantStreamEvent) -> None:
        # logger.info("In on_event of event is ", event.event, flush=True)

        if event.event == "thread.run.requires_action":
            logger.info("\nthread.run.requires_action > submit tool call")
            logger.info(f"ARGS: {self.arguments}")
