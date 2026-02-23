from langchain_core.documents import Document

chunk = Document(
    page_content=('정한 보장개시일 이전에 발생한 질병에<br>대하여 계약을 무효로 하는 경우에도 제2조(특별면책조건의<br>내용) 제1항에서 정한 '
 '특정질병에 대하여 면책을 조건으로<br>체결한 후 보장개시일 이전에 동일한 특정질병이 발생한 경<br>우에는 계약을 무효로 하지 '
 "않습니다.</p><h1 id='85' style='font-size:18px'>제2조(특별면책조건의 내용)</h1><br><p "
 "id='86' data-category='paragraph' style='font-size:16px'>\uf000 이 특별약관에서 정한 "
 '회사가 보험금을 지급하지'),
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
 'indexing': {'chunk_id': 'chunk_000833',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
