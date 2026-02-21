from langchain_core.documents import Document

chunk = Document(
    page_content=('이 경우 그 대표자는 각각 다른 계약자 또는 보험수익자를 대리하는 것으로 합니다.\n'
 '② 지정된 계약자 또는 보험수익자의 소재가 확실하지 않은 경우에는 이 특별약관에 관\n'
 '하여 회사가 계약자 또는 보험수익자 1명에 대하여 한 행위는 각각 다른 계약자 또는\n'
 '보험수익자에게도 효력이 미칩니다.\n'
 '③ 계약자가 2명 이상인 경우에는 그 책임을 연대로 합니다.- \n'
 '# <예시안내># [계약자가 2명 이상인 경우]계약자가 2명 이상인 경우 계약 전 알릴 의무, 보험료 납입의무 등 보험계약에 따른 '
 '계약자의 의무\n'
 '를 연대로 합니다.<용어풀이>[연대]'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000181',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
