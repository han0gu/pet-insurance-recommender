from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 계약자, 피보험자 또는 보험수익자는 제9조(알릴 의무 위반의 효과) 및 제2항의 보험금 지급사유조사와 관련하여 의료기관, '
 '국민건강보험공단, 경찰서 등 관공서에 대한 회 사의 서면에 의한 조사요청에 동의하여야 합니다. 다만, 정 당한 사유 없이 이에 동의하지 '
 '않을 경우 사실 확인이 끝날 때까지 회사는 보험금 지급지연에 따른 이자를 지급하지 않 습니다.\n'
 '\uf000 회사는 제5항의 서면조사에 대한 동의 요청시 조사목적,\n'
 '사용처 등을 명시하고 설명합니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 90},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000207',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
