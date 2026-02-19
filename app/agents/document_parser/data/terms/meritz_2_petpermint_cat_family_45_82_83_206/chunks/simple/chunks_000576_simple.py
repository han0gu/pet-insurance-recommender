from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 이 특별약관은 보험계약(특별약관이 부가된 경우에는 특 별약관을 포함합니다. 이하 같습니다)을 체결할 때 반려동 물의 '
 '건강상태가 회사가 정한 기준에 적합하지 않은 경우 또는 보험계약을 체결한 후 계약 전 알릴 의무 위반의 효과 등으로 보장을 제한할 경우 '
 '보험계약자(이하 「계약자」라 합니다)의 청약과 보험회사의 승낙으로 보험계약(이하 「계 약」이라 합니다)에 부가하여 이루어집니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 166},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000576',
              'chunk_char_len': 213,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
