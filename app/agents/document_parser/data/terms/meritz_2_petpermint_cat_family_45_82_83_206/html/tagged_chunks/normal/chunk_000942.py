from langchain_core.documents import Document

chunk = Document(
    page_content=("id='27' data-category='paragraph' style='font-size:16px'>2) 평형기능의 장해는 장해판정 "
 '직전 1년 이상 지속적인<br>치료 후 장해가 고착되었을 때 판정하며, 뇌병변 여<br>부, 전정기능 이상 및 장해상태를 평가하기 위해 '
 "아<br>래의 검사들을 기초로 한다.</p><br><p id='28' data-category='list' "
 "style='font-size:20px'>가) 뇌영상검사(CT, MRI)<br>나) 온도안진검사, 전기안진검사(또는 비디오안진검사) "
 '등</p><footer'),
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
 'indexing': {'chunk_id': 'chunk_000942',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
