from langchain_core.documents import Document

chunk = Document(
    page_content=('. 5. 피보험자: 반려묘의 소유와 관련하여 보험사고로 손해를 입은 사람을 말합니다. 6. 반려묘 : 보험증권에 기재된 반려묘를 말하며, '
 '이 특별약관에서 가입 가능한 반려 묘는 대한민국 내에서 피보험자와 거주를 함께하고 있는 고양이(猫)를 말합니다. 다만 아래에 기재된 '
 '고양이(猫)는 이 보험의 가입 대상이 아닙니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 96},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000523',
              'chunk_char_len': 177,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
