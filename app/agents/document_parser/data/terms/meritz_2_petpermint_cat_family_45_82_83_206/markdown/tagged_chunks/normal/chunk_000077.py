from langchain_core.documents import Document

chunk = Document(
    page_content=('자적 형태의 정보를 말한다.가. 서명자의 신원\n'
 '나. 서명자가 해당 전자문서에 서명하였다는 사실\uf000 제3항에도 불구하고 전화를 이용하여 계약을 체결하는\n'
 '경우 다음의 각 호의 어느 하나를 충족하는 때에는 자필서\n'
 '명을 생략할 수 있으며, 제2항의 규정에 따른 음성녹음 내\n'
 '용을 문서화한 확인서를 계약자에게 드림으로써 계약자 보\n'
 '관용 청약서를 전달한 것으로 봅니다.- ① 계약자, 피보험자 및 보험수익자가 동일한 계약의 경\n'
 '- 우\n'
 '- ② 계약자, 피보험자가 동일하고 보험수익자가 계약자의\n'
 '- 법정상속인인 계약일 경우'),
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
 'indexing': {'chunk_id': 'chunk_000077',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
