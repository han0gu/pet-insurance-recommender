from langchain_core.documents import Document

chunk = Document(
    page_content=('- 1) 한 팔의 3대 관절 중 관절 하나에 기능장해가 생기고 다른 관절 하나에 기능장\n'
 '- 해가 발생한 경우 지급률은 각각 적용하여 합산한다.\n'
 '- 2) 1상지(팔과 손가락)의 후유장해지급률은 원칙적으로 각각 합산하되, 지급률은\n'
 '- 60% 한도로 한다.\n'
 '- 9. 다리의 장해\n'
 '# 가. 장해의 분류장 해 의 분 류 지급률(%)\n'
 '1) 두 다리의 발목 이상을 잃었을 때 100\n'
 '2) 한 다리의 발목 이상을 잃었을 때 60\n'
 '3) 한 다리의 3대 관절 중 관절 하나의 기능을 완전히 잃었을 때 30- 143 --'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000796',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
