from langchain_core.documents import Document

chunk = Document(
    page_content=('비용손해 관련 특별약관을 체결할 때 이 특별<br>약관에서 정한 피보험자 및 반려동물의 나이에 미달되었거<br>나 초과되었을 경우에는 '
 '계약을 무효로 하며 이미 납입한<br>보험료를 돌려드립니다'),
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
 'indexing': {'chunk_id': 'chunk_000348',
              'chunk_char_len': 107,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
