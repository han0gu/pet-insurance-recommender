from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 장해분 류표의 각 신체부위별 판정기준에서 별도로 정한 경우에는 그 기준에 따릅니다. ⑦ 이미 이 특별약관에서 상해 '
 '후유장해보험금 지급사유에 해당되지 않았거나(보장개시 이전의 원인에 의하거나 또는 그 이전에 발생한 후유장해를 포함합니다), 상해 후유장 '
 '해보험금이 지급되지 않았던 피보험자에게 그 신체의 동일 부위에 또다시 제6항에 규 정하는 후유장해상태가 발생하였을 경우에는 직전까지의 '
 '후유장해에 대한 상해 후유 장해보험금이 지급된 것으로 보고 최종 후유장해 상태에 해당되는 상해 후유장해보험 금에서 이를 차감하여 '
 '지급합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 68},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000351',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
