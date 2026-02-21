from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 변경된 보험수익<br>자가 회사에 권리를 대항하기 위해서는 계약자가 보험수익<br>자가 변경되었음을 회사에 통지하여야 '
 "합니다.</p><footer id='16' style='font-size:14px'>68</footer><h1 id='17' "
 "style='font-size:20px'>【부가설명】</h1><br><p id='18' data-category='paragraph' "
 "style='font-size:20px'>계약자가 보험수익자가 변경되었음을 회사에 통지하기 전<br>에 보험금 지급사유가 발생한 경우 "
 '회사는 변경 전'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000161',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
