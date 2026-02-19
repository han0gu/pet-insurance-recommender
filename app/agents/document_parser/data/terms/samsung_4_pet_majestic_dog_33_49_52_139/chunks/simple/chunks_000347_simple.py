from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 장해분류표에 장해판정시기를 별 도로 정한 경우에는 그에 따릅니다. ② 제1항에 따라 장해지급률이 결정되었으나 그 이후 보장받을 '
 '수 있는 기간(계약의 효 력이 없어진 경우에는 보험기간이 10년 이상인 계약은 상해 발생일부터 2년 이내로 하고, 보험기간이 10년 '
 '미만인 계약은 상해 발생일부터 1년 이내)에 장해상태가 더 악 화된 때에는 그 악화된 장해상태를 기준으로 장해지급률을 결정합니다. ③ '
 '장해분류표에 해당되지 않는 후유장해는 피보험자의 직업, 연령, 신분 또는 성별 등에'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 69},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000347',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
