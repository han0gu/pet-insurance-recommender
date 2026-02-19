from langchain_core.documents import Document

chunk = Document(
    page_content=('① 제3조(보험금의 지급사유) 에서 장해지급률이 상해 발생일부터 180일 이내에 확정되 지 않는 경우에는 상해 발생일부터 180일이 되는 '
 '날의 의사진단에 기초하여 고정될 것으로 인정되는 상태를 장해지급률로 결정합니다. 다만, 장해분류표에 장해판정시기 를 별도로 정한 경우에는 '
 '그에 따릅니다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 34},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000012',
              'chunk_char_len': 161,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
