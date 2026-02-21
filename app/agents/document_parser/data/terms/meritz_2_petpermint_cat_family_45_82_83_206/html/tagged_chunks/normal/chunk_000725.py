from langchain_core.documents import Document

chunk = Document(
    page_content=('회사는 다음에 정한 사유 중 하나에 의해 피보험자가 부<br>담한 치료비, 비용 또는 손해에 대해서는 보험금을 지급하<br>지 '
 "않습니다.</p><br><p id='34' data-category='list' style='font-size:16px'>① 반려동물의 "
 '선천적, 유전적 질병에 의한 손해(보험계<br>약 이전부터 객관적으로 인지할 수 있는 증상을 포함<br>합니다'),
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
 'indexing': {'chunk_id': 'chunk_000725',
              'chunk_char_len': 208,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
