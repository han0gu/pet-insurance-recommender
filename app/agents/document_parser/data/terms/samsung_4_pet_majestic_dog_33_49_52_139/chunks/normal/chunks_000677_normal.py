from langchain_core.documents import Document

chunk = Document(
    page_content=('. ⑤ 제1항에도 불구하고 농림축산식품부장관에게 신고한 축산농장에 상시고용된 수의사와「동물원 및 수족관의 관리에 관한 법률」 제8조에 '
 '따라 허가받은 동물원 또는 수족관에 상시고용된 수 의사는 해당 농장, 동물원 또는 수족관의 동물에게 투여할 목적으로 처방대상 동물용 '
 '의약품에 대한 처방전을 발급할 수 있다. 이 경우 상시고용된 수의사의 범위, 신고방법, 처방전 발급 및 보존 방법, 진료부 작성 및 '
 '보고, 교육, 준수사항 등 그 밖에 필요한 사항은 농림축산식품부령으 로 정한다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 113},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000677',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
