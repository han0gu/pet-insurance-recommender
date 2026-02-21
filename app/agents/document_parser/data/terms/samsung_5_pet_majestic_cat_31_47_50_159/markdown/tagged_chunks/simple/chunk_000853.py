from langchain_core.documents import Document

chunk = Document(
    page_content=('| 옷입고 벗기 | · 상 · 하의 의복 착탈시 다른 사람의 계속적인 도움이 필요한 상태(10%) · 상 · 하의 의복 착탈시 부분적으로 '
 '다른 사람의 도움이 필요한 상태 또는 상의 또 는 하의중 하나만 혼자서 착탈의가 가능한 상태(5%) · 상 · 하의 의복 착탈시 혼자서 '
 '가능하나 미세동작(단추 잠그고 풀기, 지퍼 올리고 내리기, 끈 묶고 풀기 등)이 필요한 마무리는 타인의 도움이 필요한 상태(3%) |\n'
 '- 149 -[별표-상해관련1] 골절 분류표약관에 규정하는 골절로 분류되는 상병은 제9차 개정 한국표준질병 ·사인분류(통계청 고'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000853',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
