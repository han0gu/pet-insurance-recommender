from langchain_core.documents import Document

chunk = Document(
    page_content=('보험자가 서면으로 질문한 사항은 중요한 사항으로 추정 한다.\n'
 '제8조(계약 후 알릴 의무)\n'
 '\uf000 계약자 또는 피보험자는 보험기간 중에 다음 각 호의 변 경이 발생한 경우에는 우편, 전화, 방문 등의 방법으로 지 체없이 '
 '회사에 알려야 합니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 91},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000213',
              'chunk_char_len': 132,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
