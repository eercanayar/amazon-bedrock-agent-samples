import copy
import inspect
import json
from datetime import datetime
from typing import Any, Callable, Dict, Union
from termcolor import colored

from InlineAgent.constants import TraceColor


class ProcessROC:
    @staticmethod
    async def process_roc(
        inlineSessionState: Dict, roc_event: Dict, tool_map: Dict[str, Callable],
        trace_callback: Callable[[str, str, str], None] = None,
    ):
        print("========== PROCESS_ROC STARTED ==========")
        print(f"ROC Event: {json.dumps(roc_event, indent=2)}")
        print(f"Has trace_callback: {trace_callback is not None}")
        # TODO: Tool to invoke is str and callable
        if "returnControlInvocationResults" in inlineSessionState:
            raise ValueError(
                "returnControlInvocationResults key is not supported in sessionState"
            )

        if "invocationId" in inlineSessionState:
            raise ValueError("invocationId key is not supported in sessionState")

        inlineSessionState = copy.deepcopy(inlineSessionState)
        inlineSessionState = {"returnControlInvocationResults": []}
        inlineSessionState["invocationId"] = roc_event["invocationId"]
        print(f"Processing invocation ID: {roc_event['invocationId']}")

        for invocationInput in roc_event["invocationInputs"]:

            # This is a Tagged Union structure. Only one of the following top level keys will be set: apiInvocationInput, functionInvocationInput.
            # If a client receives an unknown member it will set SDK_UNKNOWN_MEMBER as the top level key, which maps to the name or tag of the unknown member.
            # The structure of SDK_UNKNOWN_MEMBER is as follows: 'SDK_UNKNOWN_MEMBER': {'name': 'UnknownMemberName'}
            if "apiInvocationInput" in invocationInput:
                raise ValueError(
                    "apiInvocationInput is not supported in returnControlInvocationResults"
                )

            actionInvocationType = invocationInput["functionInvocationInput"][
                "actionInvocationType"
            ]
            functionInvocationInput = invocationInput["functionInvocationInput"]
            actionGroup = functionInvocationInput["actionGroup"]
            
            print(f"Processing function: {functionInvocationInput['function']} from action group: {actionGroup}")
            print(f"Action invocation type: {actionInvocationType}")

            parameters = dict()
            for param in functionInvocationInput["parameters"]:
                if param["type"] == "array":
                    result = None
                    original_value = param["value"]
                    # Remove control characters that might cause JSON parsing to fail
                    cleaned_value = ''.join(ch for ch in original_value if ord(ch) >= 32 or ch in '\n\r\t')
                    
                    try:
                        result = json.loads(cleaned_value)
                    except Exception as e1:
                        try:
                            json_str = (
                                cleaned_value
                                .replace("=", ":")
                                .replace("[{", '[{"')
                                .replace("}]", '"}]')
                            )
                            json_str = json_str.replace(", ", '", "').replace(":", '":"')
                            result = json.loads(json_str)
                        except Exception as e2:
                            # If JSON parsing fails even after cleanup attempts, return an error message
                            error_msg = f"Failed to parse tool parameter '{param['name']}'. The model provided an invalid JSON format: {str(e2)}"
                            print(colored(f"JSON parsing error: {error_msg}", TraceColor.invocation_input))
                            return {
                                "returnControlInvocationResults": [{
                                    "functionResult": {
                                        "actionGroup": functionInvocationInput["actionGroup"],
                                        "agentId": functionInvocationInput["agentId"],
                                        "function": functionInvocationInput["function"],
                                        "responseBody": {"TEXT": {"body": error_msg}},
                                        "responseState": "FAILURE"
                                    }
                                }]
                            }
                    
                    parameters[param["name"]] = result
                elif param["type"] == "string":
                    parameters[param["name"]] = param["value"]
                elif param["type"] == "number":
                    parameters[param["name"]] = int(param["value"])
                elif param["type"] == "boolean":
                    parameters[param["name"]] = bool(param["value"])
                elif param["type"] == "integer":
                    parameters[param["name"]] = int(param["value"])
            if (
                actionInvocationType == "RESULT"
                or actionInvocationType == "USER_CONFIRMATION_AND_RESULT"
            ):
                tool_to_invoke: Callable = None
                if functionInvocationInput["function"] in tool_map:
                    tool_to_invoke = tool_map[functionInvocationInput["function"]]
                    print(f"Found tool to invoke: {tool_to_invoke.__name__}")
                else:
                    print(f"WARNING: Tool {functionInvocationInput['function']} not found in tool_map!")

                if not tool_to_invoke:
                    raise ValueError(
                        f"Function {functionInvocationInput['function']} not found in tools or tools class"
                    )

                if actionInvocationType == "USER_CONFIRMATION_AND_RESULT":
                    await ProcessROC.process_user_confirmation(
                        sessionState=inlineSessionState,
                        tool_to_invoke=tool_to_invoke,
                        functionInvocationInput=functionInvocationInput,
                        include_result=True,
                        parameters=parameters,
                        trace_callback=trace_callback,
                    )

                else:
                    print(f"Invoking tool {functionInvocationInput['function']} with parameters: {json.dumps(parameters)}")
                    function_result = await ProcessROC.invoke_roc_function(
                        functionInvocationInput=functionInvocationInput,
                        tool_to_invoke=tool_to_invoke,
                        parameters=parameters,
                        confirm=None,
                        trace_callback=trace_callback,
                    )
                    print(f"Function result: {json.dumps(function_result)}")
                    inlineSessionState["returnControlInvocationResults"].append(
                        {
                            "functionResult": function_result
                        }
                    )

            elif actionInvocationType == "USER_CONFIRMATION":
                tool_to_invoke = functionInvocationInput["function"]
                await ProcessROC.process_user_confirmation(
                    sessionState=inlineSessionState,
                    tool_to_invoke=tool_to_invoke,
                    functionInvocationInput=functionInvocationInput,
                    include_result=False,
                    parameters=parameters,
                    trace_callback=trace_callback,
                )

        inlineSessionState.update(inlineSessionState)
        
        print(f"Final session state: {json.dumps(inlineSessionState)}")
        print("========== PROCESS_ROC COMPLETED ==========")
        return inlineSessionState

    @staticmethod
    async def process_user_confirmation(
        sessionState: Dict,
        functionInvocationInput: Dict,
        include_result: bool,
        parameters: Dict,
        tool_to_invoke: Union[str, Callable] = None,
        trace_callback: Callable[[str, str, str], None] = None,
    ):
        while True:
            if isinstance(tool_to_invoke, Callable):
                tool_name = tool_to_invoke.__name__
            else:
                tool_name = tool_to_invoke
            confirmation_message = f"Do you want to proceed with {tool_name} with parameters : {json.dumps(parameters)}?"
            response = input(f"{confirmation_message} (y/n): ").lower()
            if response in ["y", "yes"]:
                if include_result:
                    sessionState["returnControlInvocationResults"].append(
                        {
                            "functionResult": await ProcessROC.invoke_roc_function(
                                functionInvocationInput=functionInvocationInput,
                                tool_to_invoke=tool_to_invoke,
                                confirm="CONFIRM",
                                parameters=parameters,
                                trace_callback=trace_callback,
                            )
                        }
                    )
                else:
                    sessionState["returnControlInvocationResults"].append(
                        {
                            "functionResult": {
                                "actionGroup": functionInvocationInput["actionGroup"],
                                "agentId": functionInvocationInput["agentId"],
                                "function": functionInvocationInput["function"],
                                "confirmationState": "CONFIRM",
                            }
                        }
                    )
                break
            elif response in ["n", "no"]:
                if include_result:
                    sessionState["returnControlInvocationResults"].append(
                        {
                            "functionResult": {
                                "actionGroup": functionInvocationInput["actionGroup"],
                                "agentId": functionInvocationInput["agentId"],
                                "function": functionInvocationInput["function"],
                                "responseBody": {
                                    "TEXT": {
                                        "body": "Access Denied to this function. Do not try again."
                                    }
                                },
                                "confirmationState": "DENY",
                                # "responseState": "FAILURE"
                            }
                        }
                    )
                else:
                    sessionState["returnControlInvocationResults"].append(
                        {
                            "functionResult": {
                                "actionGroup": functionInvocationInput["actionGroup"],
                                "agentId": functionInvocationInput["agentId"],
                                "function": functionInvocationInput["function"],
                                "confirmationState": "DENY",
                                # "responseState": "FAILURE"
                            }
                        }
                    )
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")

    @staticmethod
    async def invoke_roc_function(
        functionInvocationInput: Dict,
        parameters: Dict = dict(),
        confirm: str = None,
        tool_to_invoke: Callable = None,
        trace_callback: Callable[[str, str, str], None] = None,
    ) -> Dict:
        print("========== INVOKE_ROC_FUNCTION STARTED ==========")
        print(f"Function: {functionInvocationInput['function']}")
        print(f"Parameters: {json.dumps(parameters)}")
        print(f"Has trace_callback: {trace_callback is not None}")
        functionResult = dict

        # TODO: responseState
        try:

            print(f"Executing tool: {tool_to_invoke.__name__}")
            if inspect.iscoroutinefunction(tool_to_invoke):
                print("Tool is async, awaiting result...")
                result = await tool_to_invoke(**parameters)
            else:
                print("Tool is sync, executing directly...")
                result = tool_to_invoke(**parameters)
            
            # Enhanced logging for tool results
            print(f"\n{'*'*20} TOOL EXECUTION RESULT {'*'*20}")
            print(f"Tool: {tool_to_invoke.__name__}")
            print(f"Raw result: {result}")
            print(f"Result type: {type(result)}")
            
            # Format the result for better visibility
            formatted_result = None
            if isinstance(result, dict):
                try:
                    formatted_result = json.dumps(result, indent=2)
                    print(f"Result as formatted JSON:\n{formatted_result}")
                except Exception as e:
                    print(f"Error formatting result as JSON: {e}")
                    formatted_result = str(result)
            else:
                formatted_result = str(result)
                
            print(f"{'*'*60}\n")
            
            # Create a standardized tool output message
            tool_output_msg = f"Tool output result: {formatted_result}"
            print(
                colored(
                    tool_output_msg,
                    TraceColor.invocation_input,
                )
            )
            
            # Trigger trace callback for tool output with enhanced data
            if trace_callback:
                print("Invoking trace callback for tool output.")
                
                # Send a standard trace message
                trace_callback(tool_output_msg, "invocation_output", json.dumps({"tool_output": result}))
                
                # Send a dedicated tool_output trace with more detailed information
                tool_trace_data = {
                    "tool_name": tool_to_invoke.__name__,
                    "parameters": parameters,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Format the trace message to be clearly visible in logs
                detailed_trace_msg = f"TOOL OUTPUT: {tool_to_invoke.__name__} returned {formatted_result}"
                trace_callback(detailed_trace_msg, "tool_output", json.dumps(tool_trace_data))
                
                print(f"Sent detailed tool output trace with type 'tool_output'")
            else:
                print("No trace callback provided for tool output.")
                
            functionResult = {
                "actionGroup": functionInvocationInput["actionGroup"],
                "agentId": functionInvocationInput["agentId"],
                "function": functionInvocationInput["function"],
                "responseBody": {"TEXT": {"body": result}},
            }
        except Exception as e:
            functionResult = {
                "actionGroup": functionInvocationInput["actionGroup"],
                "agentId": functionInvocationInput["agentId"],
                "function": functionInvocationInput["function"],
                "responseBody": {"TEXT": {"body": e}},
                "responseState": "FAILURE",
            }

        if confirm:
            if confirm == "CONFIRM":
                functionResult["confirmationState"] = confirm
                print(f"Returning function result with confirmation: {json.dumps(functionResult)}")
                print("========== INVOKE_ROC_FUNCTION COMPLETED ==========")
                return functionResult
            else:
                print(f"Invalid confirmation value: {confirm}")
                print("========== INVOKE_ROC_FUNCTION ERROR ==========")
                raise ValueError("Only CONFIRM is a value value")
        else:
            print(f"Returning function result: {json.dumps(functionResult)}")
            print("========== INVOKE_ROC_FUNCTION COMPLETED ==========")
            return functionResult
