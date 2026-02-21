from langchain_core.documents import Document

chunk = Document(
    page_content=('을 위하여「개인정보 보호법」,「신용정보의 이용 및 보호에 관한 법률」등 관계 법\n'
 '령에 정한 경우를 제외하고 계약자, 피보험자 또는 보험수익자의 동의없이 수집, 이\n'
 '용, 조회 또는 제공하지 않습니다. 다만, 회사는 이 특별약관의 체결, 유지, 보험금 지'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000286',
              'chunk_char_len': 140,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
