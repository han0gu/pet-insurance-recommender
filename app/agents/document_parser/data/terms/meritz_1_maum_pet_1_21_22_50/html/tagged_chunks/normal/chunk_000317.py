from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 제3<br>종 단체는 구성원이 명확하고 위험의 동질성이 확보되어야 합니다.</p><footer id='63' "
 "style='font-size:14px'>- 37 -</footer><p id='64' data-category='paragraph' "
 "style='font-size:14px'>② 단체 구성원의 일부만을 대상으로 가입하는 경우에는 대상단체의 위험과 피보험단체의<br>위험의 "
 "동질성이 유지되어야 합니다.</p><h1 id='65' style='font-size:14px'>제4조(보험의 목적의 증가 감소 또는"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000317',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
