from langchain_core.documents import Document

chunk = Document(
    page_content=('② 피보험자가 보험기간 중 사망하고, 그 후에「특정법정감염병」을 직접적인 원인으로 사망한 사실이 확인된 경우에는 그 사망일을 진단 '
 '확정일로 보고 제1조(보험금의 지 급사유)에 해당하는 경우에 한하여 해당 보험금을 지급합니다. 다만, 제5조(특별약관 의 소멸)에 따라 '
 '이 특별약관의 계약자적립액 및 미경과보험료를 지급한 경우에는, 이 미 지급된 계약자적립액 및 미경과보험료를 차감하고 그 차액을 '
 '지급합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 88},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000479',
              'chunk_char_len': 226,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
