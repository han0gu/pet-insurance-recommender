from langchain_core.documents import Document

chunk = Document(
    page_content=('. ③ 이 특별약관은 피보험자가 이륜자동차를 소유, 사용(직업, 직무 또는 동호회 활동과 출퇴근용도 등으로 주로 사용하는 경우에 한하며 '
 '일회적인 사용은 제외), 관리하는 경 우에 한하여 부가하여 이루어 집니다. ④ 보험계약이 해지, 기타사유에 의하여 효력이 없게 된 '
 '경우에는 이 특별약관도 더 이상 효력이 없습니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 132},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000825',
              'chunk_char_len': 176,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
