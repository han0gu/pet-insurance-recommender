from langchain_core.documents import Document

chunk = Document(
    page_content=('음식물 섭취 | · 입으로 식사를 전혀 할 수 없어 계속적으로 튜브(비위관 또는 위루관)나 경정맥 수액을 통해 부분 혹은 전적인 '
 '영양공급을 받는 상태(20%) · 수저 사용이 불가능하여 다른 사람의 계속적인 도움이 없이는 식사를 전혀 할 수 없는 상태(15%) · '
 '숟가락 사용은 가능하나 젓가락 사용이 불가능하여 음식물 섭취에 있어 부분적으 로 다른 사람의 도움이 필요한 상태(10%) · 독립적인 '
 '음식물 섭취는 가능하나 젓가락을 이용하여 생선을 바르거나 음식물을 자르지는 못하는 상태(5%)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 149},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive', 'other']},
 'indexing': {'chunk_id': 'chunk_000988',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
