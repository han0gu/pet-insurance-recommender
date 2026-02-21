from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 회사가 나이의 착오를 발견<br>하였을 때 이미 계약나이에 도달한 경우에는 유효한 계약으<br>로 봅니다.<br>\uf000 '
 '회사의 고의 또는 과실로 계약이 무효로 된 경우와 회사<br>가 승낙 전에 무효임을 알았거나 알 수 있었음에도 보험료<br>를 반환하지 '
 '않은 경우에는 보험료를 납입한 날의 다음날부<br>터 반환일까지의 기간에 대하여 회사는 보험계약대출이율을<br>연단위 복리로 계산한 '
 "금액을 더하여 돌려 드립니다.</p><h1 id='97' style='font-size:20px'>제13조(계약내용의 변경"),
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
 'indexing': {'chunk_id': 'chunk_000349',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
