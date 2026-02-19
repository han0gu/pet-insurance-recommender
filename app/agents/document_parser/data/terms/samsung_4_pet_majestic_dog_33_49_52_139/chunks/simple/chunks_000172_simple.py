from langchain_core.documents import Document

chunk = Document(
    page_content=('제46조 (준거법)\n'
 '이 계약은 대한민국 법에 따라 규율되고 해석되며, 약관에서 정하지 않은 사항은 ｢금융소 비자 보호에 관한 법률｣, 상법, 민법 등 관계 '
 '법령을 따릅니다.\n'
 '제 47조 (예금보험에 의한 지급보장)\n'
 '회사가 파산 등으로 인하여 보험금 등을 지급하지 못할 경우에는 예금자보호법에서 정하 는 바에 따라 그 지급을 보장합니다.\n'
 '<용어풀이>\n'
 '[예금자보호제도]'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 48},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000172',
              'chunk_char_len': 203,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
