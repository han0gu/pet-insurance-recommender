from langchain_core.documents import Document

chunk = Document(
    page_content=('- 류표의 각 신체부위별 판정기준에서 별도로 정한 경우에는 그 기준에 따릅니다.\n'
 '- ⑦ 이미 이 특별약관에서 상해 후유장해보험금 지급사유에 해당되지 않았거나(보장개시\n'
 '- 이전의 원인에 의하거나 또는 그 이전에 발생한 후유장해를 포함합니다), 상해 후유장\n'
 '- 해보험금이 지급되지 않았던 피보험자에게 그 신체의 동일 부위에 또다시 제6항에 규\n'
 '- 정하는 후유장해상태가 발생하였을 경우에는 직전까지의 후유장해에 대한 상해 후유\n'
 '- 장해보험금이 지급된 것으로 보고 최종 후유장해 상태에 해당되는 상해 후유장해보험'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000297',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
