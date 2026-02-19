from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 때 단체보험의 보험수익자를 피보험자 또는 그 상속인이 아닌 자로 지정할 때에는 단체의 규약에서 명시적으로 정한 경우가 아니면 이를 '
 '적용합 니다. 2. 만 15세 미만자, 심신상실자 또는 심신박약자를 피보험자로 하여 사망을 보험금 지 급사유로 한 경우. 다만, '
 '심신박약자가 계약을 체결하거나 소속 단체의 규약에 따 라 단체보험의 피보험자가 될 때에 의사능력이 있는 경우 계약이 유효합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 40},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000104',
              'chunk_char_len': 220,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
