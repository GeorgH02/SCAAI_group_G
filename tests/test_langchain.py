"""Real LangChain schemas/messages, fake model and SQL boundary; no services."""

import importlib.util
import unittest
from unittest.mock import Mock, patch

AVAILABLE = importlib.util.find_spec('langchain_core') is not None
if AVAILABLE:
    from langchain_core.messages import AIMessage
    from langchain_agent.tools import build_tools
    from langchain_agent.workflow import EvidenceWorkflow, WorkflowError, local_model, shield_names


@unittest.skipUnless(AVAILABLE, 'Install requirements.txt for LangChain tests')
class LangChainTests(unittest.TestCase):
    def workflow(self, responses, **kwargs):
        self.model = Mock()
        self.model.bind_tools.return_value = self.model
        self.model.invoke.side_effect = responses
        return EvidenceWorkflow(object(), self.model,
                                mask=lambda s: s.replace('private@example.com', '<EMAIL>'),
                                input_guard=kwargs.pop('input_guard', lambda _: True),
                                evidence_guard=kwargs.pop('evidence_guard', lambda _: True), **kwargs)

    def call(self, name='search_listings', args=None):
        return AIMessage('', tool_calls=[{'name': name, 'args': args or {}, 'id': 'call_1'}])

    def test_all_adapters_preserve_evidence(self):
        arguments = [{}, {'listing_ids': [1], 'check_in': '2026-12-31', 'check_out': '2027-01-01'},
                     {'district': 'Neubau'}, {'host_id': 7}, {'query': 'quiet'}]
        connection = object()
        for tool, args in zip(build_tools(connection), arguments):
            evidence = {'evidence': [{'sql': 'SELECT %s', 'parameters': [1]}], 'rows': []}
            with patch('langchain_agent.queries.' + tool.name, return_value=evidence) as query:
                self.assertEqual(tool.invoke(args), evidence)
                expected = {k: v for k, v in tool.args_schema(**args).model_dump().items()
                            if v is not None or k not in ('since', 'until')}
                query.assert_called_once_with(connection, **expected)

    def test_schema_rejects_unsafe_or_mistyped_inputs(self):
        tool = build_tools(object())[0]
        for args in ({'sql': 'DELETE'}, {'limit': 101}, {'listing_ids': []},
                     {'superhost': 'true'}, {'accommodates': True}, {'offset': -1}):
            with self.assertRaises(ValueError), patch('langchain_agent.queries.search_listings') as query:
                tool.invoke(args)
            query.assert_not_called()

    def test_masking_and_multihop_keep_original_artifact(self):
        artifact = {'rows': [{'listing_id': 1, 'name': 'private@example.com'}],
                    'evidence': [{'sql': 'SELECT %s', 'parameters': [1]}]}
        flow = self.workflow([self.call(), self.call('search_reviews', {'query': 'quiet', 'listing_ids': [1]}),
                              AIMessage('UNVERIFIED CLAIM')])
        with patch('langchain_agent.queries.search_listings', return_value=artifact), \
                patch('langchain_agent.queries.search_reviews', return_value={'rows': []}) as review:
            result = flow.invoke('private@example.com apartment')
        self.assertEqual(result['status'], 'evidence_ready')
        self.assertEqual(result['results'][0], artifact)
        review.assert_called_once()
        self.assertEqual(review.call_args.kwargs['query'], 'quiet')
        self.assertEqual(review.call_args.kwargs['listing_ids'], [1])
        messages = self.model.invoke.call_args.args[0]
        self.assertNotIn('private@example.com', str(messages))
        self.assertNotIn('UNVERIFIED CLAIM', str(result))

    def test_input_block_prevents_model_and_database(self):
        flow = self.workflow([], input_guard=lambda _: False)
        self.assertEqual(flow.invoke('off topic'), {'status': 'blocked', 'results': []})
        self.model.invoke.assert_not_called()

    def test_cleaned_evidence_replaces_the_result_and_the_model_context(self):
        artifact = {'rows': [{'listing_id': 1, 'name': 'Nice'}, {'listing_id': 2, 'name': 'HOSTILE TEXT'}],
                    'evidence': [{'sql': 'SELECT %s', 'parameters': [1]}]}
        cleaned = {**artifact, 'rows': artifact['rows'][:1]}
        calls = []
        flow = self.workflow([self.call(), AIMessage('')], clean_evidence=lambda r: calls.append(r) or cleaned)
        with patch('langchain_agent.queries.search_listings', return_value=artifact):
            result = flow.invoke('apartments')
        self.assertEqual(calls, [artifact])
        self.assertEqual(result['results'], [cleaned])
        self.assertNotIn('HOSTILE TEXT', str(self.model.invoke.call_args.args[0]))

    def test_clean_evidence_must_be_callable(self):
        with self.assertRaises(ValueError):
            self.workflow([], clean_evidence='not callable')

    def test_indirect_block_prevents_second_inference(self):
        flow = self.workflow([self.call()], evidence_guard=lambda _: False)
        with patch('langchain_agent.queries.search_listings', return_value={'rows': ['injection']}):
            self.assertEqual(flow.invoke('apartments')['status'], 'blocked_evidence')
        self.assertEqual(self.model.invoke.call_count, 1)

    def test_unknown_tool_and_backend_errors_are_safe(self):
        flow = self.workflow([self.call('execute_sql')])
        with self.assertRaises(WorkflowError):
            flow.invoke('apartments')
        flow = self.workflow([self.call()])
        with patch('langchain_agent.queries.search_listings', side_effect=RuntimeError('SECRET DSN')):
            with self.assertRaises(WorkflowError) as caught:
                flow.invoke('apartments')
        self.assertNotIn('SECRET', str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_round_call_and_context_budgets(self):
        flow = self.workflow([self.call()], max_rounds=1)
        with patch('langchain_agent.queries.search_listings', return_value={'rows': []}):
            self.assertEqual(flow.invoke('apartments')['status'], 'budget_exhausted')
        message = AIMessage('', tool_calls=[{'name': 'search_listings', 'args': {}, 'id': str(i)}
                                           for i in range(2)])
        flow = self.workflow([message], max_calls=1)
        with patch('langchain_agent.queries.search_listings') as query:
            self.assertEqual(flow.invoke('apartments')['status'], 'budget_exhausted')
            query.assert_not_called()
        flow = self.workflow([self.call()], max_context_chars=1)
        with patch('langchain_agent.queries.search_listings', return_value={'rows': []}):
            self.assertEqual(flow.invoke('apartments')['status'], 'context_limit')
        self.model.invoke.assert_not_called()

    def test_context_overflow_stops_before_followup_inference(self):
        flow = self.workflow([self.call()], max_context_chars=2000)
        with patch('langchain_agent.queries.search_listings', return_value={'rows': ['x' * 3000]}):
            self.assertEqual(flow.invoke('apartments')['status'], 'context_limit')
        self.assertEqual(self.model.invoke.call_count, 1)

    def test_mask_failure_and_malformed_call_stop(self):
        flow = self.workflow([])
        flow.mask = lambda _: None
        with self.assertRaises(WorkflowError):
            flow.invoke('apartments')
        self.model.invoke.assert_not_called()
        response = AIMessage('', invalid_tool_calls=[{
            'name': 'search_listings', 'args': '{bad json', 'id': 'bad', 'error': 'parse',
        }])
        flow = self.workflow([response])
        with self.assertRaises(WorkflowError):
            flow.invoke('apartments')

    def test_no_evidence_and_request_isolation(self):
        flow = self.workflow([AIMessage('invented fact'), AIMessage('other invented fact')])
        self.assertEqual(flow.invoke('apartments'), {'status': 'no_evidence', 'results': []})
        self.assertEqual(flow.invoke('reviews'), {'status': 'no_evidence', 'results': []})
        self.assertEqual(len(self.model.invoke.call_args.args[0]), 2)

    def test_local_model_rejects_remote_endpoints(self):
        for url in ('https://api.openai.com/v1', 'http://example.com/v1',
                    'http://user:password@localhost/v1', 'http://localhost/v1?key=secret'):
            with self.assertRaises(ValueError):
                local_model(model='local', base_url=url)

    def test_null_dates_from_the_model_use_the_default_window(self):
        tool = next(t for t in build_tools(object()) if t.name == 'search_reviews')
        with patch('langchain_agent.queries.search_reviews', return_value={'rows': []}) as query:
            tool.invoke({'query': 'quiet', 'since': None, 'until': None})
        self.assertNotIn('since', query.call_args.kwargs)
        self.assertNotIn('until', query.call_args.kwargs)

    def test_local_model_timeout_is_bounded_and_applied(self):
        for bad in (0, -1, 901, True, 60.5, '60'):
            with self.assertRaises(ValueError):
                local_model(timeout=bad)
        self.assertEqual(local_model().request_timeout, 60)
        self.assertEqual(local_model(timeout=300).request_timeout, 300)

    def test_schemas_convert_to_model_tool_definitions(self):
        from langchain_core.utils.function_calling import convert_to_openai_tool
        definitions = [convert_to_openai_tool(tool) for tool in build_tools(object())]
        self.assertEqual(len(definitions), 5)
        for definition in definitions:
            self.assertNotIn('connection', definition['function']['parameters']['properties'])
            self.assertEqual(definition['type'], 'function')

    def test_environment_tracing_is_disabled(self):
        from langsmith.run_helpers import get_tracing_context
        flow = self.workflow([])
        def invoke(*args, **kwargs):
            self.assertFalse(get_tracing_context()['enabled'])
            return AIMessage('discarded')
        self.model.invoke.side_effect = invoke
        with patch.dict('os.environ', {'LANGSMITH_TRACING': 'true'}):
            self.assertEqual(flow.invoke('apartments')['status'], 'no_evidence')

    @unittest.skipUnless(importlib.util.find_spec('langchain_openai'), 'Requires local transport package')
    def test_local_transport_constructs_without_network(self):
        import asyncio
        model = local_model()
        self.assertEqual(model.model_name, 'qwen2.5:7b')
        self.assertEqual(str(model.openai_api_base), 'http://127.0.0.1:21434/v1')
        model.http_client.close()
        asyncio.run(model.http_async_client.aclose())

    @unittest.skipUnless(importlib.util.find_spec('langchain_openai'), 'Requires local transport package')
    def test_ollama_transport_preserves_tool_call_round_trip(self):
        import asyncio
        import json
        import httpx
        from langchain_core.messages import HumanMessage, ToolMessage
        from openai import OpenAI

        requests = []
        def respond(request):
            self.assertEqual(str(request.url), 'http://127.0.0.1:21434/v1/chat/completions')
            body = json.loads(request.content)
            requests.append(body)
            self.assertEqual(body['model'], 'qwen2.5:7b')
            self.assertEqual(body['tools'][0]['function']['name'], 'search_listings')
            message = {'role': 'assistant', 'content': 'Evidence collected.'}
            if len(requests) == 1:
                message = {'role': 'assistant', 'content': '', 'tool_calls': [{
                    'id': 'ollama_call_1', 'type': 'function',
                    'function': {'name': 'search_listings', 'arguments': '{"district":"Neubau"}'},
                }]}
            return httpx.Response(200, json={
                'id': 'chatcmpl-local', 'object': 'chat.completion', 'created': 0,
                'model': 'qwen2.5:7b', 'choices': [{
                    'index': 0, 'message': message,
                    'finish_reason': 'tool_calls' if len(requests) == 1 else 'stop',
                }],
            })

        model = local_model()
        model.http_client.close()
        model.http_client = httpx.Client(transport=httpx.MockTransport(respond))
        model.root_client = OpenAI(
            api_key='local-only', base_url=str(model.openai_api_base),
            http_client=model.http_client,
        )
        model.client = model.root_client.chat.completions
        try:
            agent = model.bind_tools(build_tools(object()))
            messages = [HumanMessage('Find listings in Neubau')]
            response = agent.invoke(messages)
            self.assertEqual(response.tool_calls[0]['args'], {'district': 'Neubau'})
            messages.extend([response, ToolMessage(content='{"rows": []}',
                                                   tool_call_id=response.tool_calls[0]['id'])])
            self.assertEqual(agent.invoke(messages).content, 'Evidence collected.')
            self.assertEqual(requests[1]['messages'][-1]['role'], 'tool')
            self.assertEqual(requests[1]['messages'][-1]['tool_call_id'], 'ollama_call_1')
        finally:
            model.http_client.close()
            asyncio.run(model.http_async_client.aclose())


@unittest.skipUnless(AVAILABLE, 'Install requirements.txt for LangChain tests')
class ShieldNamesTests(unittest.TestCase):
    @staticmethod
    def mask(text):  # fake masker that reads place names as people
        for word in ('Albertina', 'Wien Mitte', 'Anna'):
            text = text.replace(word, '<PERSON_1>')
        return text

    def test_listed_names_survive_but_other_text_is_still_masked(self):
        names = ['Albertina', 'Wien Mitte', 'Wien']
        shielded = shield_names(self.mask, names)
        self.assertEqual(shielded('Albertina'), 'Albertina')
        self.assertEqual(shielded('near Wien Mitte and Albertina, ask Anna'), 'near Wien Mitte and Albertina, ask <PERSON_1>')
        self.assertEqual(self.mask('Albertina'), '<PERSON_1>')  # unshielded, the masker changes it

    def test_matching_is_case_insensitive_whole_word_and_keeps_original_text(self):
        shielded = shield_names(self.mask, ['Albertina'])
        self.assertEqual(shielded('albertina'), 'albertina')
        self.assertEqual(shielded('Albertinas Anna'), '<PERSON_1>s <PERSON_1>')  # no whole-word match, so it is masked

    def test_longest_name_wins_and_empty_list_is_passthrough(self):
        shielded = shield_names(self.mask, ['Wien', 'Wien Mitte'])
        self.assertEqual(shielded('Wien Mitte'), 'Wien Mitte')
        self.assertIs(shield_names(self.mask, []), self.mask)
        self.assertIs(shield_names(self.mask, [' ', None]), self.mask)

    def test_a_mangled_token_loses_the_name_instead_of_leaking_text(self):
        shielded = shield_names(lambda text: '<PERSON_1>', ['Albertina'])
        self.assertEqual(shielded('Albertina'), '<PERSON_1>')


if __name__ == '__main__':
    unittest.main()
