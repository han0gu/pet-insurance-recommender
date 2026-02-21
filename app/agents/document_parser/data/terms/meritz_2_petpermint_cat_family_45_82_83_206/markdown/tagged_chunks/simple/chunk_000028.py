from langchain_core.documents import Document

chunk = Document(
    page_content=('이자는 더하여 지급하지 않습니다. 다만, 회사는 계약자 등\n'
 '이 분쟁조정을 신청했다는 사유만으로 이자지급을 거절하지\n'
 '않습니다.\n'
 '\uf000 계약자, 피보험자 또는 보험수익자는 제17조(알릴 의무\n'
 '위반의 효과) 및 제2항의 보험금 지급사유조사와 관련하여\n'
 '의료기관, 국민건강보험공단, 경찰서 등 관공서에 대한 회\n'
 '사의 서면에 의한 조사요청에 동의하여야 합니다. 다만, 정\n'
 '당한 사유 없이 이에 동의하지 않을 경우 사실확인이 끝날\n'
 '때까지 회사는 보험금지급지연에 따른 이자를 지급하지 않\n'
 '습니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000028',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
