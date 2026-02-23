from langchain_core.documents import Document

chunk = Document(
    page_content=("말한다.</p><br><p id='48' data-category='paragraph' style='font-size:16px'>나) "
 '치매의 장해평가는 임상적인 증상 뿐 아니라 뇌영<br>상검사(CT 및 MRI, SPECT 등)를 기초로 진단되어<br>져야 하며, '
 '18개월 이상 지속적인 치료 후 평가한<br>다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_001101',
              'chunk_char_len': 174,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
