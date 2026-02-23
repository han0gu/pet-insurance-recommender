from langchain_core.documents import Document

chunk = Document(
    page_content=('지급합니다. 다만, 장해분류표의 각 신체부위별 판정기준에 별도로 정한 경우에는 그\n'
 '기준에 따릅니다.\n'
 '⑥ 다른 상해로 인하여 후유장해가 2회 이상 발생하였을 경우에는 그 때마다 이에 해당\n'
 '하는 후유장해지급률을 결정합니다. 그러나 그 후유장해가 이미 상해 후유장해보험금\n'
 '을 지급받은 동일한 부위에 가중된 때에는 최종 장해상태에 해당하는 상해 후유장해\n'
 '보험금에서 이미 지급받은 상해 후유장해보험금을 차감하여 지급합니다. 다만, 장해분\n'
 '류표의 각 신체부위별 판정기준에서 별도로 정한 경우에는 그 기준에 따릅니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000296',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
