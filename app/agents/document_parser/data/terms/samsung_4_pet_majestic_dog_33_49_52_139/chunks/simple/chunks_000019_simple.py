from langchain_core.documents import Document

chunk = Document(
    page_content=('. ⑦ 이미 이 계약에서 상해 후유장해(80%이상)보험금 지급사유에 해당되지 않았거나(보장 개시 이전의 원인에 의하거나 또는 그 이전에 '
 '발생한 후유장해를 포함합니다), 상해 후유장해(80%이상)보험금이 지급되지 않았던 피보험자에게 그 신체의 동일 부위에 또다시 제6항에 '
 '규정하는 후유장해상태가 발생하였을 경우에는 직전까지의 후유장해 에 대한 상해 후유장해(80%이상)보험금이 지급된 것으로 보고 최종 '
 '후유장해 상태에 해당되는 상해 후유장해(80%이상)보험금에서 이를 차감하여 지급합니다.'),
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
 'indexing': {'chunk_id': 'chunk_000019',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
