from langchain_core.documents import Document

chunk = Document(
    page_content=('사망진단서, 장해진단<br>서, 입원치료확인서, 의사처방전(처방조제비) 등)<br>③ 신분증(주민등록증이나 운전면허증 등 사진이 붙은 '
 '정<br>부기관발행 신분증, 본인이 아닌 경우에는 본인의 인<br>감증명서, 본인서명사실확인서 또는 안전성과 신뢰성<br>이 확보된 '
 '전자적 수단을 활용한 보험수익자 의사표시<br>의 확인방법 포함)<br>④ 기타 보험수익자가 보험금의 수령에 필요하여 제출하<br>는 '
 "서류</p><br><p id='55' data-category='paragraph' style='font-size:16px'>\uf000 "
 '제1항'),
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
 'indexing': {'chunk_id': 'chunk_000040',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
