from langchain_core.documents import Document

chunk = Document(
    page_content=('- 부기관발행 신분증, 본인이 아닌 경우에는 본인의 인\n'
 '- 감증명서, 본인서명사실확인서 또는 안전성과 신뢰성\n'
 '- 이 확보된 전자적 수단을 활용한 보험수익자 의사표시\n'
 '- 의 확인방법 포함)\n'
 '- ④ 기타 보험수익자가 보험금의 수령에 필요하여 제출하\n'
 '- 는 서류\n'
 '\uf000 제1항 제2호의 사고증명서는 수의사법 제12조(진단서\n'
 '등)에서 규정한 내용에 따라 국내의 동물병원에서 수의사에\n'
 '의해 발급한 것이어야 합니다.# 【수의사법 제12조(진단서 등)】- ① 수의사는 자기가 직접 진료하거나 검안하지 아니하고'),
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
 'indexing': {'chunk_id': 'chunk_000155',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
