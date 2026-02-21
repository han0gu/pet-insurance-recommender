from langchain_core.documents import Document

chunk = Document(
    page_content=('도움이 필요한 상태(3%)</td></tr><tr><td>옷입고 벗기</td><td>- 상, 하의 의복 착탈시 다른 사람의 계속적인 '
 '도움이 필요한 상태(10%) - 상, 하의 의복 착탈시 부분적으로 다른 사람의 도움이 필요한 상태 또는 상의 또는 하의중 하 나만 혼자서 '
 '착탈의가 가능한 상태(5%) - 상, 하의 의복착탈시 혼자서 가능하나 미세동 작(단추 잠그고 풀기, 지퍼 올리고 내리기, 끈 묶고 풀기 '
 '등)이 필요한 마무리는 타인의 도움 이 필요한 상태(3%)</td></tr></tbody></table><footer'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001114',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
