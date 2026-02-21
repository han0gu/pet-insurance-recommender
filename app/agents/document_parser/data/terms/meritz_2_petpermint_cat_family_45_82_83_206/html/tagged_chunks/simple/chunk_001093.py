from langchain_core.documents import Document

chunk = Document(
    page_content=('않은 경우에는 그로써 고정되거나 중하게<br>된 장해에 대해서는 인정하지 않는다.<br>다) “정신행동에 극심한 장해를 남긴 때”라 함은 '
 '장<br>해판정 직전 1년 이상 지속적인 정신건강의학과의<br>치료를 받았으며 GAF 30점 이하인 상태를 말한다.<br>라) '
 '“정신행동에 심한 장해를 남긴 때”라 함은 장해<br>판정 직전 1년 이상 지속적인 정신건강의학과의 치<br>료를 받았으며 GAF 40점 '
 '이하인 상태를 말한다.<br>마) “정신행동에 뚜렷한 장해를 남긴 때”라 함은 장<br>해판정 직전 1년 이상 지속적인'),
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
 'indexing': {'chunk_id': 'chunk_001093',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
