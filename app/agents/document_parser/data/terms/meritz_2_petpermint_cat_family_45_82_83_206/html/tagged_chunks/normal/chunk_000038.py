from langchain_core.documents import Document

chunk = Document(
    page_content=("직무로 하는 사람이 직무상 선<br>박에 탑승하고 있는 동안</p><h1 id='50' "
 "style='font-size:20px'>제6조(보험금 지급사유의 통지)</h1><br><p id='51' "
 "data-category='paragraph' style='font-size:20px'>계약자 또는 피보험자나 보험수익자는 "
 '제3조(보험금의 지급<br>사유)에서 정한 보험금 지급사유의 발생을 안 때에는 지체<br>없이 그 사실을 회사에 알려야 '
 "합니다.</p><h1 id='52' style='font-size:20px'>제7조(보험금의"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000038',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
