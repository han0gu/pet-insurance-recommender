from langchain_core.documents import Document

chunk = Document(
    page_content=('발생할 수 있는 경제적 불안을 제 거하기 위해 공동으로 재산을 준비하여 두는 제도) 사업을 실시하는 경영 주체와 공제 계약 자 사이에 '
 "체결되는 계약을 말합니다.</td></tr></tbody></table><br><h1 id='84' "
 "style='font-size:16px'>\uf000 지급금과 이자율 관련 용어</h1><br><table id='85' "
 "style='font-size:16px'><thead><tr><td>용어</td><td>정의</td></tr></thead><tbody><tr><td>연단위 "
 '복리</td><td>회사가'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000278',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
