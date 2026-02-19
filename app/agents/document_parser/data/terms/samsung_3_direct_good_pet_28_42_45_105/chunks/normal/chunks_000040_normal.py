from langchain_core.documents import Document

chunk = Document(
    page_content=('제12조 (대표자의 지정)\n'
 '① 계약자 또는 보험수익자가 2명 이상인 경우에는 각 대표자를 1명 지정하여야 합니다. 이 경우 그 대표자는 각각 다른 계약자 또는 '
 '보험수익자를 대리하는 것으로 합니다. ② 지정된 계약자 또는 보험수익자의 소재가 확실하지 않은 경우에는 이 계약에 관하여 회사가 계약자 '
 '또는 보험수익자 1명에 대하여 한 행위는 각각 다른 계약자 또는 보험 수익자에게도 효력이 미칩니다. ③ 계약자가 2명 이상인 경우에는 그 '
 '책임을 연대로 합니다.\n'
 '<예시안내>'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 31},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000040',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
