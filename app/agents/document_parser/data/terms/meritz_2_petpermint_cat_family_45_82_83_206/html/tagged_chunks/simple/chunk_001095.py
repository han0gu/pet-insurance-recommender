from langchain_core.documents import Document

chunk = Document(
    page_content=('주) 능력장애측정기준의 항목 : ㉮ 적절한 음식<br>섭취, ㉯ 대소변관리, 세면, 목욕, 청소 등의 청<br>결 유지, ㉰ 적절한 '
 '대화기술 및 협조적인 대인<br>관계, ㉱ 규칙적인 통원․약물 복용, ㉲ 소지품 및<br>금전관리나 적절한 구매행위, ㉳ '
 "대중교통이나<br>일반공공시설의 이용</p><br><p id='43' data-category='list' "
 "style='font-size:16px'>바) “정신행동에 약간의 장해를 남긴 때”라 함은 장<br>해판정 직전 1년 이상 지속적인 "
 '정신건강의학과의<br>치료를 받았으며,'),
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
 'indexing': {'chunk_id': 'chunk_001095',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
