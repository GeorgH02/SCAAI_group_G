import re
import unittest

from guardrails.output_guard import NOTHING_VERIFIED, EvidenceIndex, OutputGuard, missing_numbers, numbers, redact_pii


CALLS = []


def word_overlap(pairs):
    """Fake NLI: a hypothesis is entailed when its words about the flat appear in the premise."""
    CALLS.extend(pairs)
    scores = []
    for premise, hypothesis in pairs:
        claims = {w for w in re.findall(r'[a-zäöüß]+', hypothesis.lower())
                  if w in {'quiet', 'clean', 'neubau', 'pool', 'rooftop', 'best', 'superhost', 'available'}}
        scores.append(1.0 if claims <= set(re.findall(r'[a-zäöüß]+', premise.lower())) else 0.0)
    return scores


def finder(pattern, kind='PERSON'):
    """Fake Masker.find: every regex match is reported as PII."""
    return lambda text: [(m.start(), m.end(), kind) for m in re.finditer(pattern, text)]


fake_find = finder(r'\b(?:Anna|Thomas)\b')


SQL = [{
    'tool': 'search_listings', 'notices': ['Prices are snapshot base prices.'],
    'rows': [{'listing_id': 101, 'neighbourhood': 'Neubau', 'accommodates': 4, 'price': '120.00',
              'host_id': 7, 'host_is_superhost': True, 'lnight_road_band': '50-54'}],
}, {
    'tool': 'compare_host_prices', 'notices': [],
    'groups': [{'host_is_superhost': True, 'listing_count': 12, 'priced_listing_count': 11,
                'missing_price_count': 1, 'average_base_price': 133.456, 'listing_ids': [101, 102]}],
}, {
    'tool': 'check_availability', 'notices': [], 'check_in': '2026-12-31', 'check_out': '2027-01-01',
    'availability': [{'listing_id': 101, 'status': 'available', 'failures': [], 'unknowns': []}],
}]
REVIEWS = [{'review_id': 9001, 'listing_id': 101, 'review_date': '2026-05-01', 'text': 'Very quiet at night and clean.'}]


class OutputGuardTests(unittest.TestCase):
    def setUp(self):
        self.guard = OutputGuard(word_overlap, find_pii=fake_find)

    def check(self, answer, **kwargs):
        return self.guard.check(answer, SQL, REVIEWS, **kwargs)

    def test_supported_sentences_are_kept_with_their_evidence(self):
        result = self.check('Listing 101 is in Neubau and accommodates 4 guests [listing 101]. '
                            'Guests say it is quiet at night [review 9001].')
        self.assertTrue(result.verified)
        self.assertEqual(result.removed, [])
        self.assertIn('[review 9001]', result.answer)
        self.assertEqual([c['citations'] for c in result.citation_map], [['listing 101'], ['review 9001']])
        self.assertIn('Very quiet at night', result.citation_map[1]['evidence'][0])

    def test_unsupported_sentence_is_removed_and_others_stay(self):
        result = self.check('Guests say it is quiet at night [review 9001]. It has a rooftop pool [review 9001].')
        self.assertEqual(result.answer, 'Guests say it is quiet at night [review 9001].')
        self.assertIn('not entailed', result.removed[0].reason)

    def test_citation_to_unretrieved_evidence_is_removed(self):
        result = self.check('Guests say it is quiet at night [review 9001]. It is quiet [review 5].')
        self.assertIn('review 5', result.removed[0].reason)
        self.assertNotIn('[review 5]', result.answer)

    def test_wrong_number_is_removed_even_if_words_match(self):
        result = self.check('Listing 101 accommodates 6 guests [listing 101]. '
                            'Listing 101 accommodates 4 guests [listing 101].')
        self.assertEqual(result.answer, 'Listing 101 accommodates 4 guests [listing 101].')
        self.assertIn('numbers', result.removed[0].reason)

    def test_rounded_average_and_dates_match_the_evidence(self):
        result = self.check('Superhost listings average €133.46 across 12 listings [listing 101]. '
                            'Listing 101 is available from 2026-12-31 [listing 101].')
        self.assertEqual(result.removed, [], [s.reason for s in result.removed])

    def test_uncited_sentences_must_restate_a_notice(self):
        result = self.check('Guests say it is quiet at night [review 9001]. Prices are snapshot base prices. '
                            'This is the best flat in Vienna.')
        self.assertIn('Prices are snapshot base prices.', result.answer)
        self.assertNotIn('best flat', result.answer)

    def test_uncited_sentence_gets_the_citation_of_the_evidence_that_supports_it(self):
        result = self.check('Guests say it is quiet at night. Listing 101 is in Neubau.')
        self.assertTrue(result.verified, [s.reason for s in result.removed])
        self.assertEqual(result.answer, 'Guests say it is quiet at night. [review 9001] Listing 101 is in Neubau. [listing 101]')
        self.assertEqual([c['citations'] for c in result.citation_map], [['review 9001'], ['listing 101']])

    def test_bare_number_citations_are_resolved(self):
        result = self.check('Listing 101 is in Neubau [101]. Guests say it is quiet at night [9001].')
        self.assertTrue(result.verified, [s.reason for s in result.removed])
        self.assertEqual(result.answer, 'Listing 101 is in Neubau [listing 101]. Guests say it is quiet at night [review 9001].')

    def test_uncited_summary_can_be_supported_by_several_pieces_of_evidence(self):
        reviews = [*REVIEWS, {'review_id': 9002, 'listing_id': 101, 'review_date': '2026-05-02', 'text': 'Superhost was great.'}]
        result = OutputGuard(word_overlap).check('Guests praise the quiet nights and the superhost.', [], reviews)
        self.assertTrue(result.verified, [s.reason for s in result.removed])
        self.assertEqual(sorted(result.citation_map[0]['citations']), ['review 9001', 'review 9002'])

    def test_uncited_comparison_combines_two_price_groups(self):
        sql = [{'tool': 'compare_host_prices', 'district': 'Leopoldstadt', 'groups': [
            {'host_is_superhost': True, 'listing_count': 3, 'average_base_price': '166.0412', 'listing_ids': [1]},
            {'host_is_superhost': False, 'listing_count': 9, 'average_base_price': '149.04', 'listing_ids': [2]}]}]
        result = OutputGuard(word_overlap).check('Superhost listings average 166.04 versus 149.04 for others.', sql, [])
        self.assertTrue(result.verified, [s.reason for s in result.removed])

    def test_the_fact_with_the_sentences_numbers_is_tried_even_among_many_similar_facts(self):
        rows = [{'listing_id': 100 + i, 'distance_m': str(300 + i), 'lnight_road_band': '70-74'} for i in range(15)]
        rows.append({'listing_id': 200, 'distance_m': '162', 'lnight_road_band': '65-69'})
        sql = [{'tool': 'search_listings', 'landmark': 'Wien Westbahnhof', 'rows': rows, 'notices': []}]
        result = OutputGuard(word_overlap).check('The nearest one is 162 m away with 65-69 dB at night.', sql, [])
        self.assertTrue(result.verified, [s.reason for s in result.removed])
        self.assertEqual(result.citation_map[0]['citations'], ['listing 200'])
        self.assertIn('closest of the listings found to Wien Westbahnhof', result.citation_map[0]['evidence'][0])
        self.assertIn('162 m from Wien Westbahnhof', result.citation_map[0]['evidence'][0])

    def test_each_notice_is_a_separate_candidate(self):
        CALLS.clear()
        sql = [{'tool': 'search_listings', 'rows': [], 'notices': ['Prices are snapshot base prices.',
                                                                   'There are no listings in Neubau below 45 dB.']}]
        OutputGuard(word_overlap).check('There are no listings in Neubau below 45 dB.', sql, [])
        self.assertIn('There are no listings in Neubau below 45 dB.', [p for p, _ in CALLS])

    def test_withheld_fallback_is_not_masked(self):
        guard = OutputGuard(word_overlap, find_pii=finder('Withheld|withheld'))
        result = guard.check('It has a rooftop pool [review 9001].', SQL, REVIEWS)
        self.assertEqual(result.answer, NOTHING_VERIFIED)
        self.assertFalse(result.pii_redacted)

    def test_numbers_from_the_evidence_survive_pii_masking(self):
        guard = OutputGuard(word_overlap, find_pii=finder(r'\d{3,}(?:\.\d+)?', 'PHONE_NUMBER'))
        result = guard.check('Listing 101 costs €120.00 and is in Neubau [listing 101]. Call 0664123456.', SQL, REVIEWS)
        self.assertIn('€120.00', result.answer)
        self.assertFalse(result.pii_redacted)

    def test_citation_after_full_stop_belongs_to_the_sentence(self):
        result = self.check('Guests say it is quiet at night. [review 9001]')
        self.assertTrue(result.verified)
        self.assertEqual(result.sentences[0].citations, [('review', '9001')])

    def test_markdown_bullets_and_headings_are_kept_only_with_content(self):
        answer = ('## Quiet listings\n- Guests say it is quiet at night [review 9001].\n'
                  '## Pools\n- It has a rooftop pool [review 9001].')
        result = self.check(answer)
        self.assertEqual(result.answer, '## Quiet listings\n- Guests say it is quiet at night [review 9001].')

    def test_nothing_verified_returns_a_fallback(self):
        result = self.check('It has a rooftop pool [review 9001].')
        self.assertFalse(result.verified)
        self.assertEqual(result.answer, NOTHING_VERIFIED)

    def test_pii_in_output_is_masked(self):
        result = self.check('Anna says it is quiet at night [review 9001].',)
        self.assertTrue(result.pii_redacted)
        self.assertIn('<PERSON_1>', result.answer)

    def test_question_numbers_may_be_restated(self):
        result = self.check('Listing 101 accommodates 4 guests within budget of €150 [listing 101].',
                            question='Apartments for 4 guests under €150')
        self.assertTrue(result.verified, [s.reason for s in result.removed])

    def test_nli_receives_combined_premise_for_multi_citation_sentences(self):
        CALLS.clear()
        result = self.check('Listing 101 is in Neubau and guests say it is quiet at night [review 9001] [listing 101].')
        self.assertTrue(result.verified, [s.reason for s in result.removed])
        self.assertTrue(any('Neubau' in p and 'Very quiet' in p for p, _ in CALLS))


class HelperTests(unittest.TestCase):
    def test_numbers_understand_separators(self):
        self.assertEqual(numbers('€1,234.50, 12,5 dB, 120'), [(1234.5, 2), (12.5, 1), (120.0, 0)])

    def test_missing_numbers_allows_rounding_only(self):
        self.assertEqual(missing_numbers('about €133', 'average 133.456'), [])
        self.assertEqual(missing_numbers('about €140', 'average 133.456'), [140.0])

    def test_redaction_keeps_citations_ids_placeholders_and_landmarks(self):
        find = finder(r'\b(?:\d{5,}|Thomas|Westbahnhof|PERSON_0)\b')
        text = 'Thomas near Westbahnhof, listing 48604277 [listing 48604277], <PERSON_0>, a@b.com'
        redacted, changed = redact_pii(text, find, keep_names=['Westbahnhof'], keep_numbers={'48604277'})
        self.assertTrue(changed)
        self.assertEqual(redacted, '<PERSON_1> near Westbahnhof, listing 48604277 [listing 48604277], <PERSON_0>, <EMAIL_ADDRESS_2>')

    def test_lowercase_words_are_never_masked_but_names_and_emails_are(self):
        find = finder(r'helpfulness|which is|Thomas')
        redacted, changed = redact_pii('Thomas praised the helpfulness, which is rare. Mail thomas.k@mail.at', find)
        self.assertEqual(redacted, '<PERSON_1> praised the helpfulness, which is rare. Mail <EMAIL_ADDRESS_2>')
        self.assertTrue(changed)
        self.assertEqual(redact_pii('Guests praise the helpfulness.', find), ('Guests praise the helpfulness.', False))

    def test_detection_overlapping_a_protected_number_still_masks_the_name(self):
        find = lambda text: [(text.index('Thomas'), text.index(' is'), 'PERSON')]
        redacted, _ = redact_pii('Host Thomas 9556855 is a superhost [listing 7].', find, keep_numbers={'9556855'})
        self.assertEqual(redacted, 'Host <PERSON_1> 9556855 is a superhost [listing 7].')
        phone = finder(r'\+43 660 1234567', 'PHONE_NUMBER')
        self.assertEqual(redact_pii('Call +43 660 1234567.', phone)[0], 'Call <PHONE_NUMBER_1>.')

    def test_index_links_reviews_and_rows_to_listings(self):
        index = EvidenceIndex(SQL, REVIEWS)
        premises = index.premises('listing', '101')
        self.assertTrue(any('superhost' in p for p in premises))
        self.assertTrue(any('quiet' in p for p in premises))
        self.assertEqual(index.premises('listing', '999'), [])


if __name__ == '__main__':
    unittest.main()
