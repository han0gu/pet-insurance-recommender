from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 실속형\n'
 '\uf000 회사는 보험기간 중에 보험증권에 기재된 반려동물에게 질병 또는 상해가 발생하여 그 치료를 직접적인 목적으로 수의사법 '
 '제2조(정의)에서 정한 국내 동물병원(이하 「동물'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 158},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000543',
              'chunk_char_len': 100,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
