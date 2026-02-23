from langchain_core.documents import Document

chunk = Document(
    page_content=('말<br>합니다.<br>\uf000 제1항의 자기공명영상(MRI)이라 함은 제1조(보험금의 지<br>급사유)에서 정한 수의사에 의하여 '
 '진단 및 치료가 필요하<br>다고 인정된 경우로서 수의사의 관리 하에 자기공명영상<br>(MRI)을 사용하는 촬영 의료행위를 '
 "말합니다.</p><br><p id='66' data-category='paragraph' "
 "style='font-size:20px'>\uf000 제1항의 전산화단층촬영(CT)이라 함은 제1조(보험금의<br>지급사유)에서 정한 "
 '수의사에 의하여 진단 및 치료가 필요<br>하다고 인정된 경우로서'),
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
 'indexing': {'chunk_id': 'chunk_000822',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
