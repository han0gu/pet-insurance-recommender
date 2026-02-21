from langchain_core.documents import Document

chunk = Document(
    page_content=('- 1. 피보험자가 고의로 자신을 해친 경우. 다만, 피보험자가 심신상실 등으로 자유\n'
 '- 로운 의사결정을 할 수 없는 상태에서 자신을 해친 경우에는 보험금을 지급합\n'
 '- 니다.\n'
 '- 2. 보험수익자가 고의로 피보험자를 해친 경우. 다만, 그 보험수익자가 보험금의\n'
 '- 일부 보험수익자인 경우에는 다른 보험수익자에 대한 보험금은 지급합니다.\n'
 '- 3. 계약자가 고의로 피보험자를 해친 경우\n'
 '- 4. 피보험자의 임신, 출산(제왕절개를 포함합니다), 산후기. 그러나, 회사가 보'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000016',
              'chunk_char_len': 260,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
