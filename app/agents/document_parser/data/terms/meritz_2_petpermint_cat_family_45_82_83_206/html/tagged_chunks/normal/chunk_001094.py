from langchain_core.documents import Document

chunk = Document(
    page_content=('“정신행동에 뚜렷한 장해를 남긴 때”라 함은 장<br>해판정 직전 1년 이상 지속적인 정신건강의학과의<br>치료를 받았으며, '
 '보건복지부고시「장애정도판정기<br>준」의“능력장애측정기준”주) 상 6개 항목 중 3개<br>항목 이상에서 독립적 수행이 불가능하여 타인의 '
 "도<br>움이 필요하고 GAF 50점 이하인 상태를 말한다.</p><br><p id='42' "
 "data-category='paragraph' style='font-size:20px'>※ 주) 능력장애측정기준의 항목 : ㉮ 적절한 "
 '음식<br>섭취, ㉯ 대소변관리, 세면,'),
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
 'indexing': {'chunk_id': 'chunk_001094',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
