from langchain_core.documents import Document

chunk = Document(
    page_content=('제 17조 (상해보험계약 후 알릴 의무)\n'
 '① 계약자 또는 피보험자는 보험기간 중에 피보험자에게 다음 각 호의 변경이 발생한 경 우에는 우편, 전화, 방문 등의 방법으로 지체없이 '
 '회사에 알려야 합니다.\n'
 '1. 보험증권 등에 기재된 직업 또는 직무의 변경\n'
 '가. 현재의 직업 또는 직무가 변경된 경우 나. 직업이 없는 자가 취직한 경우 다. 현재의 직업을 그만둔 경우\n'
 '<용어풀이>'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 36},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000060',
              'chunk_char_len': 208,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
