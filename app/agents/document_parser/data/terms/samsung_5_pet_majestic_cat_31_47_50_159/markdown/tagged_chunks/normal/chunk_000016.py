from langchain_core.documents import Document

chunk = Document(
    page_content=('후유장해(80%이상)보험금이 지급되지 않았던 피보험자에게 그 신체의 동일 부위에\n'
 '또다시 제6항에 규정하는 후유장해상태가 발생하였을 경우에는 직전까지의 후유장해\n'
 '에 대한 상해 후유장해(80%이상)보험금이 지급된 것으로 보고 최종 후유장해 상태에\n'
 '해당되는 상해 후유장해(80%이상)보험금에서 이를 차감하여 지급합니다.- \n'
 '# <유의사항>동일한 부위에 다른 원인으로 후유장해가 2회이상 발생한 경우'),
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
 'indexing': {'chunk_id': 'chunk_000016',
              'chunk_char_len': 219,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
