from langchain_core.documents import Document

chunk = Document(
    page_content=('⑤ 같은 상해로 두 가지 이상의 후유장해가 생긴 경우에는 후유장해 지급률을 합산하여 지급합니다. 다만, 장해분류표의 각 신체부위별 '
 '판정기준에 별도로 정한 경우에는 그 기준에 따릅니다. ⑥ 다른 상해로 인하여 후유장해가 2회 이상 발생하였을 경우에는 그 때마다 이에 '
 '해당 하는 후유장해지급률을 결정합니다. 그러나 그 후유장해가 이미 상해 후유장해보험금 을 지급받은 동일한 부위에 가중된 때에는 최종 '
 '장해상태에 해당하는 상해 후유장해 보험금에서 이미 지급받은 상해 후유장해보험금을 차감하여 지급합니다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 69},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['head', 'joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000350',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
