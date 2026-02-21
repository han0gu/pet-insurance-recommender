from langchain_core.documents import Document

chunk = Document(
    page_content=('재가입 적용대상 특별약관이 다음 각 호의 조건을 충족<br>하고 계약자가 제5항에 따라 재가입 의사를 표시한 때에는<br>이 특별약관의 '
 '제11조(보험계약의 성립) 및 보통약관 제21<br>조(약관 교부 및 설명 의무 등)를 준용하여 회사가 정한 절<br>차에 따라 계약자는 '
 '기존 계약에 이어 재가입할 수 있으며,<br>이 경우 회사는 기존 계약의 가입 이후 발생한 상해 또는<br>질병을 사유로 가입을 거절할 '
 "수 없습니다.</p><br><p id='13' data-category='list'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000363',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
