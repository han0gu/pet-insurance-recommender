from langchain_core.documents import Document

chunk = Document(
    page_content=('<유의사항>\n'
 '동일한 부위에 다른 원인으로 후유장해가 2회이상 발생한 경우 최종 장해상태에 해당하는 후유장해 보험금에서 아래 금액을 차감하여 '
 '지급합니다.\n'
 '- 이전의 후유장해로 이미 지급받은 보험금이 있는 경우 그 보험금\n'
 '- 이전의 후유장해가 보험금 지급사유에 해당되지 않은 경우라도,\n'
 '보험금이 지급되었다면 이전의 후유장해에 해당하는 보험금\n'
 '※ 보험금 지급사유에 해당되지 않은 경우란 장해의 원인이 보장개시 이전에 발생했거나 약관상 보험금을 지급하지 않는 사유에 해당하는 경우 '
 '등을 말합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 68},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000352',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
