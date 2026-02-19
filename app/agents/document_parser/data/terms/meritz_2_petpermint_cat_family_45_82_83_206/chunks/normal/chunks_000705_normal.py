from langchain_core.documents import Document

chunk = Document(
    page_content=('나) 객관적 검사(스트레스 엑스선)상 5mm 이상의 동요관 절(관절이 흔들리거나 움직이는 것)이 있는 경우 다) 근전도 검사상 불완전한 '
 '손상(incomplete injury)소견이 있으면서 도수근력검사(MMT)에 서 근력이 3등급(fair)인 경우\n'
 '11) 동요장해 평가 시에는 정상측과 환측을 비교하여 증 가된 수치로 평가한다. 12) "가관절주)이 남아 뚜렷한 장해를 남긴 때"라 '
 '함은 대퇴골에 가관절이 남은 경우 또는 경골과 종아리 뼈의 2개뼈 모두에 가관절이 남은 경우를 말한다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 195},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000705',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
