from langchain_core.documents import Document

chunk = Document(
    page_content=('· 빈번하고 불규칙한 배변으로 인해 2시간 이상 계속되는 업무를 수행하는 것이 어 려운 상태, 또는 배변, 배뇨는 독립적으로 가능하나 '
 '요실금, 변실금이 있는 때 (5%)\n'
 '목욕 | · 세안, 양치, 샤워, 목욕 등 모든 개인위생 관리시 타인의 지속적인 도움이 필요한 상태(10%) · 세안, 양치시 부분적인 '
 '도움 하에 혼자서 가능하나 목욕이나 샤워시 타인의 도움 이 필요한 상태(5%) · 세안, 양치와 같은 개인위생관리를 독립적으로 '
 '시행가능하나 목욕이나 샤워시 부 분적으로 타인의 도움이 필요한 상태(3%)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 149},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'urinary', 'other']},
 'indexing': {'chunk_id': 'chunk_000991',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
