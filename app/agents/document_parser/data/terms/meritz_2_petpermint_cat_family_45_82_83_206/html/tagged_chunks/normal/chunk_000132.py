from langchain_core.documents import Document

chunk = Document(
    page_content=('. 만약, 회사가 전자<br>우편 및 전자적 의사표시로 제공한 경우 계약자 또는 그 대<br>리인이 약관 및 계약자 보관용 청약서 등을 '
 "수신하였을 때<br>에는 해당 문서를 드린 것으로 봅니다.</p><br><p id='84' data-category='list' "
 "style='font-size:16px'>① 서면교부<br>② 우편 또는 전자우편<br>③ 휴대전화 문자메시지 또는 이에 준하는 전자적 "
 "의사표시</p><br><h1 id='85' style='font-size:16px'>【약관의 중요한 내용 예시】</h1><br><p"),
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
 'indexing': {'chunk_id': 'chunk_000132',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
