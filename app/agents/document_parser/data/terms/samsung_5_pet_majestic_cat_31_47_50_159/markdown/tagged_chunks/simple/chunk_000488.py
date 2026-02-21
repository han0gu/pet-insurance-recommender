from langchain_core.documents import Document

chunk = Document(
    page_content=('보험자가 서면으로 질문한 사항은 중요한 사항으로 추정한다.# 제12조 (계약 후 알릴 의무)① 계약자 또는 피보험자는 보험기간 중에 '
 '피보험자에게 다음 각 호의 변경이 발생한 경\n'
 '우에는 우편, 전화, 방문 등의 방법으로 지체없이 회사에 알려야 합니다.- 1. 청약서의 기재사항을 변경하고자 할 때 또는 변경이 '
 '생겼음을 알았을 때\n'
 '- 2. 이 특별약관에서 보장하는 위험과 동일한 위험을 보장하는 계약을 다른 보험자와\n'
 '- 체결하고자 할 때 또는 이와 같은 계약이 있음을 알았을 때\n'
 '- 3. 반려묘를 양도할 때'),
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
 'indexing': {'chunk_id': 'chunk_000488',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
