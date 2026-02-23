from langchain_core.documents import Document

chunk = Document(
    page_content=('6개 항목 중 2항목<br>이상에서 독립적 수행이 불가능하여 타인의 도움이<br>필요하고 GAF 70점 이하인 상태를 말한다.<br>아) '
 "지속적인 정신건강의학과의 치료란 3개월 이상 약<br>물치료가 중단되지 않았음을 의미한다.</p><footer id='44' "
 "style='font-size:14px'>202</footer><p id='45' data-category='list' "
 "style='font-size:16px'>자) 심리학적 평가보고서는 정신건강의학과 의료기관에<br>서 실시되어져야 하며, 자격을 갖춘"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001097',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
