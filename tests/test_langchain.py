"""Real LangChain schemas/messages, fake model and SQL boundary; no services."""

import importlib.util
import unittest
from unittest.mock import Mock, patch

AVAILABLE = importlib.util.find_spec('langchain_core') is not None
if AVAILABLE:
    from langchain_core.messages import AIMessage
    from langchain_tools import build_tools
    from langchain_workflow import EvidenceWorkflow, WorkflowError, local_model


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
            with patch('queries.' + tool.name, return_value=evidence) as query:
                self.assertEqual(tool.invoke(args), evidence)
                query.assert_called_once_with(connection, **tool.args_schema(**args).model_dump())

    def test_schema_rejects_unsafe_or_mistyped_inputs(self):
        tool = build_tools(object())[0]
        for args in ({'sql': 'DELETE'}, {'limit': 101}, {'listing_ids': []},
                     {'superhost': 'true'}, {'accommodates': True}, {'offset': -1}):
            with self.assertRaises(ValueError), patch('queries.search_listings') as query:
                tool.invoke(args)
            query.assert_not_called()

    def test_masking_and_multihop_keep_original_artifact(self):
        artifact = {'rows': [{'listing_id': 1, 'name': 'private@example.com'}],
                    'evidence': [{'sql': 'SELECT %s', 'parameters': [1]}]}
        flow = self.workflow([self.call(), self.call('search_reviews', {'query': 'quiet', 'listing_ids': [1]}),
                              AIMessage('UNVERIFIED CLAIM')])
        with patch('queries.search_listings', return_value=artifact), \
                patch('queries.search_reviews', return_value={'rows': []}) as review:
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

    def test_indirect_block_prevents_second_inference(self):
        flow = self.workflow([self.call()], evidence_guard=lambda _: False)
        with patch('queries.search_listings', return_value={'rows': ['injection']}):
            self.assertEqual(flow.invoke('apartments')['status'], 'blocked_evidence')
        self.assertEqual(self.model.invoke.call_count, 1)

    def test_unknown_tool_and_backend_errors_are_safe(self):
        flow = self.workflow([self.call('execute_sql')])
        with self.assertRaises(WorkflowError):
            flow.invoke('apartments')
        flow = self.workflow([self.call()])
        with patch('queries.search_listings', side_effect=RuntimeError('SECRET DSN')):
            with self.assertRaises(WorkflowError) as caught:
                flow.invoke('apartments')
        self.assertNotIn('SECRET', str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_round_call_and_context_budgets(self):
        flow = self.workflow([self.call()], max_rounds=1)
        with patch('queries.search_listings', return_value={'rows': []}):
            self.assertEqual(flow.invoke('apartments')['status'], 'budget_exhausted')
        message = AIMessage('', tool_calls=[{'name': 'search_listings', 'args': {}, 'id': str(i)}
                                           for i in range(2)])
        flow = self.workflow([message], max_calls=1)
        with patch('queries.search_listings') as query:
            self.assertEqual(flow.invoke('apartments')['status'], 'budget_exhausted')
            query.assert_not_called()
        flow = self.workflow([self.call()], max_context_chars=1)
        with patch('queries.search_listings', return_value={'rows': []}):
            self.assertEqual(flow.invoke('apartments')['status'], 'context_limit')
        self.model.invoke.assert_not_called()

    def test_context_overflow_stops_before_followup_inference(self):
        flow = self.workflow([self.call()], max_context_chars=2000)
        with patch('queries.search_listings', return_value={'rows': ['x' * 3000]}):
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
        model = local_model(model='test-local-alias')
        self.assertEqual(str(model.openai_api_base), 'http://127.0.0.1:8080/v1')
        model.http_client.close()
        asyncio.run(model.http_async_client.aclose())


if __name__ == '__main__':
    unittest.main()
