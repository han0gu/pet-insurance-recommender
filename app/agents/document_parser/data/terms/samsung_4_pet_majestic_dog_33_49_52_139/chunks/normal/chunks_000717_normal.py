from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 경우 상시고용된 수의사의 범위, 신고방법, 처방전 발급 및 보존 방법, 진료부 작성 및 보고, 교육, 준수사항 등 그 밖에 필요한 '
 '사항은 농림축산식품부령으 로 정한다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 117},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000717',
              'chunk_char_len': 97,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
