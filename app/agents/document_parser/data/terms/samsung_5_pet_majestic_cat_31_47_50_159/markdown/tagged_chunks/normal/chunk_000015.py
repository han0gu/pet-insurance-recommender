from langchain_core.documents import Document

chunk = Document(
    page_content=('이상)보험금을 지급받은 동일한 부위에 가중된 때에는 최종 장해상태에 해당하는 상\n'
 '해 후유장해(80%이상)보험금에서 이미 지급받은 상해 후유장해(80%이상)보험금을 차\n'
 '감하여 지급합니다. 다만, 장해분류표의 각 신체부위별 판정기준에서 별도로 정한 경\n'
 '우에는 그 기준에 따릅니다.\n'
 '⑦ 이미 이 계약에서 상해 후유장해(80%이상)보험금 지급사유에 해당되지 않았거나(보장\n'
 '개시 이전의 원인에 의하거나 또는 그 이전에 발생한 후유장해를 포함합니다), 상해\n'
 '후유장해(80%이상)보험금이 지급되지 않았던 피보험자에게 그 신체의 동일 부위에'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000015',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
