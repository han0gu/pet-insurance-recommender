from langchain_core.documents import Document

chunk = Document(
    page_content=('- 사) “정신행동에 경미한 장해를 남긴 때”라 함은 장\n'
 '- 해판정 직전 1년 이상 지속적인 정신건강의학과의\n'
 '- 치료를 받았으며, 보건복지부고시「장애정도판정기\n'
 '- 준」의“능력장애측정기준”상 6개 항목 중 2항목\n'
 '- 이상에서 독립적 수행이 불가능하여 타인의 도움이\n'
 '- 필요하고 GAF 70점 이하인 상태를 말한다.\n'
 '- 아) 지속적인 정신건강의학과의 치료란 3개월 이상 약\n'
 '- 물치료가 중단되지 않았음을 의미한다.\n'
 '202- 자) 심리학적 평가보고서는 정신건강의학과 의료기관에\n'
 '- 서 실시되어져야 하며, 자격을 갖춘 임상심리전문'),
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
 'indexing': {'chunk_id': 'chunk_000615',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
