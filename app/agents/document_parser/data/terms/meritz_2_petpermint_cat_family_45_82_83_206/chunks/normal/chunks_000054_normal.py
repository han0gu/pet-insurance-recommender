from langchain_core.documents import Document

chunk = Document(
    page_content=('제16조(상해보험계약 후 알릴 의무)\n'
 '\uf000 계약자 또는 피보험자는 보험기간 중에 피보험자에게 다 음 각 호의 변경이 발생한 경우에는 우편, 전화, 방문 등의 방법으로 '
 '지체없이 회사에 알려야 합니다.\n'
 '① 보험증권에 기재된 직업 또는 직무의 변경 1) 현재의 직업 또는 직무가 변경된 경우 2) 직업이 없는 자가 취직한 경우 3) 현재의 '
 '직업을 그만둔 경우'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 58},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000054',
              'chunk_char_len': 196,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
