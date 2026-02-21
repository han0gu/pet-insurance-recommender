from langchain_core.documents import Document

chunk = Document(
    page_content=('배상할 책임을 집니다.<br>\uf000 회사가 보험금 지급여부 및 지급금액에 관하여 현저하게<br>공정을 잃은 합의로 보험수익자에게 '
 "손해를 가한 경우에도<br>회사는 제2항에 따라 손해를 배상할 책임을 집니다.</p><br><h1 id='54' "
 "style='font-size:20px'>【 현저하게 공정을 잃은 합의 】</h1><br><p id='55' "
 "data-category='paragraph' style='font-size:20px'>사회통념상 일반 보통인이라면 그 같은 일을 하지 "
 '않을<br>정도로 현저하게 공정성을 잃은 것을'),
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
 'indexing': {'chunk_id': 'chunk_000255',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
