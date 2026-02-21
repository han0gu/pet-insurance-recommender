from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 진단계약의 경우 의료법 제3조(의료기관)의 규정<br>에 따른 종합병원과 병원에서 직장 또는 개인이 실시한 건<br>강진단서 '
 "사본 등 건강상태를 판단할 수 있는 자료로 건강<br>진단을 대신할 수 있습니다.</p><footer id='12' "
 "style='font-size:14px'>57</footer><h1 id='13' style='font-size:20px'>【 계약 전 "
 "알릴 의무 】</h1><br><p id='14' data-category='paragraph' "
 "style='font-size:20px'>상법"),
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
 'indexing': {'chunk_id': 'chunk_000078',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
