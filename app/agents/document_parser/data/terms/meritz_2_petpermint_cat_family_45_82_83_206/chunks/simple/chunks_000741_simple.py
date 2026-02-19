from langchain_core.documents import Document

chunk = Document(
    page_content=('바) “정신행동에 약간의 장해를 남긴 때”라 함은 장 해판정 직전 1년 이상 지속적인 정신건강의학과의 치료를 받았으며, '
 '보건복지부고시「장애정도판정기 준」의 “능력장애측정기준”상 6개 항목 중 2개 항목 이상에서 독립적 수행이 불가능하여 타인의 도움이 '
 '필요하고 GAF 60점 이하인 상태를 말한다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 202},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000741',
              'chunk_char_len': 164,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
