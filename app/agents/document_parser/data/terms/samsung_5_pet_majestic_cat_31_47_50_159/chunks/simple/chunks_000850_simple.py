from langchain_core.documents import Document

chunk = Document(
    page_content=('제3조 (지정대리청구인의 지정)\n'
 '① 계약자는 보험계약에서 정한 보험금을 직접 청구할 수 없는 특별한 사정이 있을 경우 를 대비하여 계약을 체결할 때 또는 계약체결 이후 '
 '다음 각 호의 어느 하나에 해당하 는 자 중에서 보험금의 청구대리인(2인 이내에서 지정하되, 2인 지정시 대표대리인을 지정)(이하 '
 '"지정대리청구인"이라 합니다)으로 지정할 수 있습니다. 또한, 지정대리청 구인은 제4조(지정대리청구인의 변경지정)에 의한 변경 지정 또는 '
 '보험금 청구시에도 다음 각 호의 어느 하나에 해당하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 135},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000850',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
