from enum import Enum
from typing import Dict, List, Callable
from InlineAgent.constants import Level, TraceColor
from termcolor import colored
from rich.console import Console
from rich.markdown import Markdown

import json


AGENT = {}
STEP = 1


class Trace:

    @staticmethod
    def parse_trace(
        trace: Dict,
        agentName: str,
        truncateResponse: int = None,
        trace_callback: Callable[[str, str, str], None] = None,
    ):
        input_tokens = 0
        output_tokens = 0
        llm_calls = 0

        if "sessionId" in trace:
            pass
            _ = trace["sessionId"]

        # This is a Tagged Union structure.
        # Only one of the following top level keys will be set: customOrchestrationTrace, failureTrace, guardrailTrace, orchestrationTrace, postProcessingTrace,
        # preProcessingTrace, routingClassifierTrace.
        # If a client receives an unknown member it will set SDK_UNKNOWN_MEMBER as the top level key, which maps to the name or tag of the unknown member.
        # The structure of SDK_UNKNOWN_MEMBER is as follows: 'SDK_UNKNOWN_MEMBER': {'name': 'UnknownMemberName'}

        HighLevelTrace.parse_custom_orchestration_trace(trace=trace, trace_callback=trace_callback)

        HighLevelTrace.parse_failure_trace(trace=trace, trace_callback=trace_callback)

        HighLevelTrace.guardrail_trace(trace=trace, trace_callback=trace_callback)

        orch_input_tokens, orch_output_tokens, orch_llm_calls = (
            HighLevelTrace.parse_orchestration_trace(trace=trace, agentName=agentName, trace_callback=trace_callback)
        )
        input_tokens += orch_input_tokens
        output_tokens += orch_output_tokens
        llm_calls += orch_llm_calls

        post_input_tokens, post_output_tokens, post_llm_calls = (
            HighLevelTrace.parse_post_processing_trace(trace=trace, trace_callback=trace_callback)
        )
        input_tokens += post_input_tokens
        output_tokens += post_output_tokens
        llm_calls += post_llm_calls

        pre_input_tokens, pre_output_tokens, pre_llm_calls = (
            HighLevelTrace.parse_preprocessing_trace(trace=trace, trace_callback=trace_callback)
        )
        input_tokens += pre_input_tokens
        output_tokens += pre_output_tokens
        llm_calls += pre_llm_calls

        rout_input_tokens, rout_output_tokens, rout_llm_calls = (
            HighLevelTrace.parse_routing_classifier_trace(
                trace=trace, agentName=agentName, trace_callback=trace_callback
            )
        )
        input_tokens += rout_input_tokens
        output_tokens += rout_output_tokens
        llm_calls += rout_llm_calls

        return int(input_tokens), int(output_tokens), int(llm_calls)

    @staticmethod
    def add_citation(citations: List, cite=1) -> str:

        agent_answer = str()

        cite_output = list()
        for citation in citations:
            text = citation["generatedResponsePart"]["textResponsePart"]["text"]
            retrievedReferences = str()
            uri = None

            for idx, retrievedReference in enumerate(citation["retrievedReferences"]):

                uri = retrievedReference["location"]["s3Location"]["uri"]
                kb_id = retrievedReference["metadata"][
                    "x-amz-bedrock-kb-data-source-id"
                ]

                if "content" in retrievedReference:
                    if retrievedReference["content"]["type"] == "TEXT":
                        retrievedReferences += (
                            f"[{idx + 1}] "
                            + retrievedReference["content"]["text"]
                            + "\n"
                        )
                    elif retrievedReference["content"]["type"] == "IMAGE":
                        retrievedReferences += (
                            f"[{idx + 1}] " + "Image is retrieved" + "\n"
                        )
                    elif retrievedReference["content"]["type"] == "ROW":
                        retrievedReferences += (
                            f"[{idx + 1}] "
                            + " ".join(
                                [
                                    f"column: {row['columnName']} value: {row['columnValue']}"
                                    for row in retrievedReference["content"]["row"]
                                ]
                            )
                            + "\n"
                        )

            cite_output.append(
                (f"[{cite}] S3 URI: {uri}\nKB ID: {kb_id}", retrievedReferences)
            )

            agent_answer += text
            print(colored(text, TraceColor.final_output), end="")
            if citation["retrievedReferences"]:
                print(colored(f" [{cite}]", TraceColor.error), end="")

            cite += 1

        print("\n\n")
        for output in cite_output:
            if len(output[1]):
                print(colored(output[0], TraceColor.cite))
                print(colored(output[1] + "\n", TraceColor.retrieved_references))

        return agent_answer, cite


class HighLevelTrace:

    @staticmethod
    def parse_custom_orchestration_trace(trace: Dict, trace_callback: Callable[[str, str, str], None] = None):
        if "customOrchestrationTrace" in trace:
            error_msg = f"Agent error: {trace['customOrchestrationTrace']['event']['text']}"
            print(
                colored(
                    error_msg,
                    TraceColor.custom_orchestraction_trace,
                )
            )
            if trace_callback:
                trace_callback(error_msg, "custom_orchestraction_trace", json.dumps(trace['customOrchestrationTrace']))

    @staticmethod
    def parse_failure_trace(trace: Dict, trace_callback: Callable[[str, str, str], None] = None):
        if "failureTrace" in trace:
            error_msg = f"Agent error: {trace['failureTrace']['failureReason']}"
            print(
                colored(
                    error_msg,
                    TraceColor.error,
                )
            )
            if trace_callback:
                trace_callback(error_msg, "error", json.dumps(trace['failureTrace']))

    @staticmethod
    def guardrail_trace(trace: Dict, trace_callback: Callable[[str, str, str], None] = None):
        if "guardrailTrace" in trace:
            if trace["guardrailTrace"]["action"] == "INTERVENED":
                guardrail_msg = "<--- Guardrail Intervened --->"
                print(
                    colored(
                        guardrail_msg, TraceColor.guardrail_trace
                    )
                )
                if trace_callback:
                    trace_callback(guardrail_msg, "guardrail_trace", json.dumps(trace['guardrailTrace']))
            if "inputAssessments" in trace["guardrailTrace"]:
                for inputAssessment in trace["guardrailTrace"]["inputAssessments"]:
                    guardrail_msg = "Input Guardrail"
                    print(colored(guardrail_msg, TraceColor.guardrail_trace))
                    assessment_str = json.dumps(inputAssessment, indent=2, default=str)
                    print(
                        colored(
                            assessment_str,
                            TraceColor.guardrail_trace,
                        )
                    )
                    if trace_callback:
                        trace_callback(f"{guardrail_msg}\n{assessment_str}", "guardrail_trace", json.dumps(inputAssessment))

            if "outputAssessments" in trace["guardrailTrace"]:
                for outputAssessment in trace["guardrailTrace"]["outputAssessments"]:
                    guardrail_msg = "Output Guardrail"
                    print(colored(guardrail_msg, TraceColor.guardrail_trace))
                    assessment_str = json.dumps(outputAssessment, indent=2, default=str)
                    print(
                        colored(
                            assessment_str,
                            TraceColor.guardrail_trace,
                        )
                    )
                    if trace_callback:
                        trace_callback(f"{guardrail_msg}\n{assessment_str}", "guardrail_trace", json.dumps(outputAssessment))

    @staticmethod
    def parse_orchestration_trace(trace: Dict, agentName: str, trace_callback: Callable[[str, str, str], None] = None):
        # This is a Tagged Union structure. Only one of the following top level keys will be set: invocationInput, modelInvocationInput, modelInvocationOutput, observation, rationale. If a client receives an unknown member it will set SDK_UNKNOWN_MEMBER as the top level key, which maps to the name or tag of the unknown member. The structure of SDK_UNKNOWN_MEMBER is as follows:'SDK_UNKNOWN_MEMBER': {'name': 'UnknownMemberName'}

        if "orchestrationTrace" in trace:

            RoutingAndOrchestrationTrace.parse_invocation_input(
                trace=trace["orchestrationTrace"],
                trace_callback=trace_callback
            )

            RoutingAndOrchestrationTrace.parse_model_invocation_input(
                trace=trace["orchestrationTrace"],
                trace_callback=trace_callback
            )

            input_tokens, output_tokens, llm_calls = (
                RoutingAndOrchestrationTrace.parse_model_invocation_output(
                    trace=trace["orchestrationTrace"],
                    trace_callback=trace_callback
                )
            )

            RoutingAndOrchestrationTrace.parse_observation(
                trace=trace["orchestrationTrace"],
                trace_callback=trace_callback
            )

            if "rationale" in trace["orchestrationTrace"]:

                # if SUPERVISOR in AGENT:
                #     # Sub agent
                # else:
                #     # Main agent
                #     print(colored("Supervisor Agent Invoked", TraceColor.rationale))
                thought_msg = f"Thought: {trace['orchestrationTrace']['rationale']['text']}"
                print(
                    colored(
                        thought_msg,
                        TraceColor.rationale,
                    )
                )
                if trace_callback:
                    trace_callback(thought_msg, "rationale", json.dumps(trace['orchestrationTrace']['rationale']))

            return input_tokens, output_tokens, llm_calls
        return 0, 0, 0

    @staticmethod
    def parse_preprocessing_trace(trace: Dict, trace_callback: Callable[[str, str, str], None] = None):

        if "preProcessingTrace" in trace:
            if "modelInvocationOutput" in trace["preProcessingTrace"]:
                input_tokens = int(
                    trace["preProcessingTrace"]["modelInvocationOutput"]["metadata"][
                        "usage"
                    ]["inputTokens"]
                )

                output_tokens = int(
                    trace["preProcessingTrace"]["modelInvocationOutput"]["metadata"][
                        "usage"
                    ]["outputTokens"]
                )

                llm_calls = 1

                pre_processing_msg = "Pre-processing trace, agent came up with an initial plan."
                stats_msg = f"Input Tokens: {input_tokens} Output Tokens: {output_tokens}"
                print(
                    colored(
                        pre_processing_msg,
                        TraceColor.pre_processing,
                    )
                )
                print(
                    colored(
                        stats_msg,
                        TraceColor.stats,
                    )
                )
                if trace_callback:
                    trace_callback(pre_processing_msg, "pre_processing", json.dumps(trace['preProcessingTrace']))
                    trace_callback(stats_msg, "stats", json.dumps({"inputTokens": input_tokens, "outputTokens": output_tokens}))

                return input_tokens, output_tokens, llm_calls
        return 0, 0, 0

    @staticmethod
    def parse_post_processing_trace(trace: Dict, trace_callback: Callable[[str, str, str], None] = None):

        if "postProcessingTrace" in trace:
            if "modelInvocationOutput" in trace["postProcessingTrace"]:
                input_tokens = int(
                    trace["postProcessingTrace"]["modelInvocationOutput"]["metadata"][
                        "usage"
                    ]["inputTokens"]
                )

                output_tokens = int(
                    trace["postProcessingTrace"]["modelInvocationOutput"]["metadata"][
                        "usage"
                    ]["outputTokens"]
                )

                llm_calls = 1
                post_processing_msg = "Agent post-processing complete."
                stats_msg = f"Input Tokens: {input_tokens} Output Tokens: {output_tokens}"
                print(
                    colored(
                        post_processing_msg,
                        TraceColor.post_processing
                    )
                )
                print(
                    colored(
                        stats_msg,
                        TraceColor.stats,
                    )
                )
                if trace_callback:
                    trace_callback(post_processing_msg, "post_processing", json.dumps(trace['postProcessingTrace']))
                    trace_callback(stats_msg, "stats", json.dumps({"inputTokens": input_tokens, "outputTokens": output_tokens}))

                return input_tokens, output_tokens, llm_calls
        return 0, 0, 0

    @staticmethod
    def parse_routing_classifier_trace(trace: Dict, agentName: str, trace_callback: Callable[[str, str, str], None] = None):
        # This is a Tagged Union structure. Only one of the following top level keys will be set: invocationInput, modelInvocationInput, modelInvocationOutput, observation. If a client receives an unknown member it will set SDK_UNKNOWN_MEMBER as the top level key, which maps to the name or tag of the unknown member. The structure of SDK_UNKNOWN_MEMBER is as follows: 'SDK_UNKNOWN_MEMBER': {'name': 'UnknownMemberName'}

        if "routingClassifierTrace" in trace:
            RoutingAndOrchestrationTrace.parse_invocation_input(
                trace=trace["routingClassifierTrace"],
                trace_callback=trace_callback
            )

            RoutingAndOrchestrationTrace.parse_model_invocation_input(
                trace=trace["routingClassifierTrace"],
                trace_callback=trace_callback
            )

            input_tokens, output_tokens, llm_calls = (
                RoutingAndOrchestrationTrace.parse_model_invocation_output(
                    trace=trace["routingClassifierTrace"],
                    trace_callback=trace_callback
                )
            )

            RoutingAndOrchestrationTrace.parse_observation(
                trace=trace["routingClassifierTrace"],
                trace_callback=trace_callback
            )

            return input_tokens, output_tokens, llm_calls
        return 0, 0, 0


class RoutingAndOrchestrationTrace:

    @staticmethod
    def parse_invocation_input(trace, trace_callback: Callable[[str, str, str], None] = None):
        if "invocationInput" in trace:
            # NOTE: when agent determines invocations should happen in parallel
            # the trace objects for invocation input still come back one at a time.
            # if "invocationType" in trace["orchestrationTrace"]["invocationInput"]:

            if "actionGroupInvocationInput" in trace["invocationInput"]:
                if "function" in trace["invocationInput"]["actionGroupInvocationInput"]:
                    tool = trace["invocationInput"]["actionGroupInvocationInput"][
                        "function"
                    ]
                elif (
                    "apiPath" in trace["invocationInput"]["actionGroupInvocationInput"]
                ):
                    tool = trace["invocationInput"]["actionGroupInvocationInput"][
                        "apiPath"
                    ]
                else:
                    tool = "undefined"

                params_info = []
                for parameter in trace["invocationInput"]["actionGroupInvocationInput"][
                    "parameters"
                ]:
                    param_str = f"{parameter['name']}[{parameter['value']}] ({parameter['type']})"
                    params_info.append(param_str)

                print(
                    colored(
                        f"Tool use: {tool} with these inputs: {' '.join(params_info)}",
                        TraceColor.invocation_input,
                    )
                )

            if "agentCollaboratorInvocationInput" in trace["invocationInput"]:
                if (
                    "input"
                    in trace["invocationInput"]["agentCollaboratorInvocationInput"]
                ):
                    text = str()
                    if (
                        "returnControlResults"
                        in trace["invocationInput"]["agentCollaboratorInvocationInput"][
                            "input"
                        ]
                    ):
                        for returnControlInvocationResult in trace["invocationInput"][
                            "agentCollaboratorInvocationInput"
                        ]["input"]["returnControlResults"][
                            "returnControlInvocationResults"
                        ]:
                            if "apiResult" in returnControlInvocationResult:
                                text += f"{returnControlInvocationResult['apiResult']['actionGroup']} :: {returnControlInvocationResult['apiResult']['apiPath']} ({returnControlInvocationResult['apiResult']['responseBody']['string']['body']})"
                            elif "functionResult" in returnControlInvocationResult:
                                text += f"{returnControlInvocationResult['functionResult']['actionGroup']} :: {returnControlInvocationResult['functionResult']['function']} ({returnControlInvocationResult['functionResult']['responseBody']['string']['body']})"

                    if text:
                        print(
                            colored(
                                f"Agent collaborator: {trace['invocationInput']['agentCollaboratorInvocationInput']['agentCollaboratorName']} invoked with {text}",
                                TraceColor.invocation_input,
                            )
                        )
                    if (
                        "text"
                        in trace["invocationInput"]["agentCollaboratorInvocationInput"][
                            "input"
                        ]
                    ):
                        text = trace["invocationInput"][
                            "agentCollaboratorInvocationInput"
                        ]["input"]["text"]
                        print(
                            colored(
                                f"Agent collaborator: {trace['invocationInput']['agentCollaboratorInvocationInput']['agentCollaboratorName']} invoked with {text}",
                                TraceColor.invocation_input,
                            )
                        )
                    else:
                        text = str()

            if "codeInterpreterInvocationInput" in trace["invocationInput"]:
                if "code" in trace["invocationInput"]["codeInterpreterInvocationInput"]:
                    code_interpreter_msg = "Code interpreter:"
                    print(colored(code_interpreter_msg, TraceColor.invocation_input))
                    if trace_callback:
                        trace_callback(code_interpreter_msg, "invocation_input", json.dumps(trace['invocationInput']['codeInterpreterInvocationInput']))
                    console = Console()
                    console.print(
                        Markdown(
                            f"**Generated code**\n```python\n{trace['invocationInput']['codeInterpreterInvocationInput']['code']}\n```"
                        )
                    )

                if (
                    "files"
                    in trace["invocationInput"]["codeInterpreterInvocationInput"]
                ):
                    print(
                        colored(
                            "Code Interpreter invoked with uploaded files",
                            TraceColor.invocation_input,
                        )
                    )

            if "knowledgeBaseLookupInput" in trace["invocationInput"]:
                print(
                    colored(
                        f"Knowledgebase retrieval: Knowledgebase Id ({trace['invocationInput']['knowledgeBaseLookupInput']['knowledgeBaseId']}) query ({trace['invocationInput']['knowledgeBaseLookupInput']['text']})",
                        TraceColor.invocation_input,
                    )
                )

    @staticmethod
    def parse_model_invocation_input(trace, trace_callback: Callable[[str, str, str], None] = None):
        if "modelInvocationInput" in trace:
            if trace["modelInvocationInput"]["type"] == "ROUTING_CLASSIFIER":
                routing_msg = "Routing the request to collaborators"
                print(
                    colored(
                        routing_msg,
                        TraceColor.rationale,
                    )
                )
                if trace_callback:
                    trace_callback(routing_msg, "rationale", json.dumps(trace))

    @staticmethod
    def parse_model_invocation_output(trace, trace_callback: Callable[[str, str, str], None] = None):

        if "modelInvocationOutput" in trace:
            if "inputTokens" in trace["modelInvocationOutput"]["metadata"]["usage"]:
                input_tokens = int(
                    trace["modelInvocationOutput"]["metadata"]["usage"]["inputTokens"]
                )
            else:
                input_tokens = 0
            if "outputTokens" in trace["modelInvocationOutput"]["metadata"]["usage"]:
                output_tokens = int(
                    trace["modelInvocationOutput"]["metadata"]["usage"]["outputTokens"]
                )
            else:
                output_tokens = 0
            llm_calls = 1
            
            # Extract and print the model's thinking from rawResponse if available
            if "rawResponse" in trace["modelInvocationOutput"]:
                try:
                    raw_response = trace["modelInvocationOutput"]["rawResponse"]
                    if isinstance(raw_response, dict) and "content" in raw_response:
                        response_content = raw_response["content"]
                        response_data = json.loads(response_content)
                        
                        if "output" in response_data and "message" in response_data["output"]:
                            message = response_data["output"]["message"]
                            if "content" in message and isinstance(message["content"], list):
                                for content_item in message["content"]:
                                    if content_item.get("text") and content_item["text"] is not None:
                                        print(colored(f"Model thinking: {content_item['text']}", TraceColor.rationale))
                except Exception as e:
                    print(colored(f"Error parsing model thinking: {e}", TraceColor.error))
            
            stats_msg = f"Input Tokens: {input_tokens} Output Tokens: {output_tokens}"
            print(
                colored(
                    stats_msg,
                    TraceColor.stats,
                )
            )
            if trace_callback:
                trace_callback(stats_msg, "stats", json.dumps({"inputTokens": input_tokens, "outputTokens": output_tokens}))
            return input_tokens, output_tokens, llm_calls
        return 0, 0, 0

    @staticmethod
    def parse_observation(trace, trace_callback: Callable[[str, str, str], None] = None):

        if "observation" in trace:

            if "actionGroupInvocationOutput" in trace["observation"]:
                tool_output_msg = f"Tool use output: {trace['observation']['actionGroupInvocationOutput']['text']}"
                print(
                    colored(
                        tool_output_msg,
                        TraceColor.invocation_output,
                    )
                )
                if trace_callback:
                    trace_callback(tool_output_msg, "invocation_output", json.dumps(trace['observation']['actionGroupInvocationOutput']))

            if "agentCollaboratorInvocationOutput" in trace["observation"]:
                if (
                    "output"
                    in trace["observation"]["agentCollaboratorInvocationOutput"]
                ):
                    if (
                        "returnControlPayload"
                        in trace["observation"]["agentCollaboratorInvocationOutput"][
                            "output"
                        ]
                    ):
                        text = str()
                        for invocationInput in trace["observation"][
                            "agentCollaboratorInvocationOutput"
                        ]["output"]["invocationInputs"]:
                            if "apiInvocationInput" in invocationInput:
                                text += f"{invocationInput['apiInvocationInput']['actionGroup']} :: {invocationInput['apiInvocationInput']['apiPath']}"
                            elif "functionInvocationInput" in invocationInput:
                                text += f"{invocationInput['functionInvocationInput']['actionGroup']} :: {invocationInput['functionInvocationInput']['function']}"

                        collab_msg = f"Agent collaborator: {trace['invocationInput']['agentCollaboratorInvocationInput']['agentCollaboratorName']} invoked with {text}"
                        print(
                            colored(
                                collab_msg,
                                TraceColor.invocation_input,
                            )
                        )
                        if trace_callback:
                            trace_callback(collab_msg, "invocation_input", json.dumps(trace['invocationInput']['agentCollaboratorInvocationInput']))
                    elif (
                        "text"
                        in trace["observation"]["agentCollaboratorInvocationOutput"][
                            "output"
                        ]
                    ):
                        text = trace["observation"][
                            "agentCollaboratorInvocationOutput"
                        ]["output"]["text"]
                        print(
                            colored(
                                f"Collaborator output: {text}",
                                TraceColor.invocation_input,
                            )
                        )
                    else:
                        text = str()

            if "codeInterpreterInvocationOutput" in trace["observation"]:
                if (
                    "executionOutput"
                    in trace["observation"]["codeInterpreterInvocationOutput"]
                ):
                    code_output_msg = f"Code interpreter output: {trace['observation']['codeInterpreterInvocationOutput']['executionOutput']}"
                    print(
                        colored(
                            code_output_msg,
                            TraceColor.invocation_output,
                        )
                    )
                    if trace_callback:
                        trace_callback(code_output_msg, "invocation_output", json.dumps(trace['observation']['codeInterpreterInvocationOutput']))

                if (
                    "executionError"
                    in trace["observation"]["codeInterpreterInvocationOutput"]
                ):
                    code_error_msg = f"Code interpreter output error: {trace['observation']['codeInterpreterInvocationOutput']['executionError']}"
                    print(
                        colored(
                            code_error_msg,
                            TraceColor.error,
                        )
                    )
                    if trace_callback:
                        trace_callback(code_error_msg, "error", json.dumps(trace['observation']['codeInterpreterInvocationOutput']))

                if (
                    "executionTimeout"
                    in trace["observation"]["codeInterpreterInvocationOutput"]
                ):
                    if trace["observation"]["codeInterpreterInvocationOutput"][
                        "executionTimeout"
                    ]:
                        print(
                            colored(
                                f"Code interpreter output error: Execution timeout",
                                TraceColor.error,
                            )
                        )

                if "files" in trace["observation"]["codeInterpreterInvocationOutput"]:
                    print(
                        colored(
                            "Code Interpreter created new files",
                            TraceColor.invocation_input,
                        )
                    )

            if "finalResponse" in trace["observation"]:
                pass

            if "knowledgeBaseLookupOutput" in trace["observation"]:
                if (
                    "retrievedReferences"
                    in trace["observation"]["knowledgeBaseLookupOutput"]
                ):
                    for retrievedReference in trace["observation"][
                        "knowledgeBaseLookupOutput"
                    ]["retrievedReferences"]:
                        if "content" in retrievedReference:
                            # TODO: ["content"]["type"] does not exist
                            # if retrievedReference["content"]["type"] == "TEXT":
                            print(
                                colored(
                                    retrievedReference["content"]["text"],
                                    TraceColor.invocation_output,
                                )
                            )
                            # elif retrievedReference["content"]["type"] == "IMAGE":
                            #     print(
                            #         colored(
                            #             "Image Retrieved", TraceColor.invocation_output
                            #         )
                            #     )
                            # elif retrievedReference["content"]["type"] == "ROW":
                            #     print(
                            #         colored(
                            #             "Row retrieved: "
                            #             + " ".join(
                            #                 [
                            #                     f"column: {row['columnName']} value: {row['columnValue']}"
                            #                     for row in retrievedReference[
                            #                         "content"
                            #                     ]["row"]
                            #                 ]
                            #             ),
                            #             TraceColor.invocation_output,
                            #         )
                            #     )

                        if "location" in retrievedReference:
                            print(
                                colored(
                                    f"Location: {json.dumps(retrievedReference['location'], indent=2, default=str)}",
                                    TraceColor.invocation_output,
                                )
                            )

            if "repromptResponse" in trace["observation"]:
                print(
                    colored(
                        f"Reprompting {trace['observation']['repromptResponse']['source']} with query {trace['orchestrationTrace']['observation']['repromptResponse']['text']}",
                        TraceColor.invocation_output,
                    )
                )
