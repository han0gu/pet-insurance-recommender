from langchain_core.documents import Document

chunk = Document(
    page_content=('등)이외는 섭취하지 못하는 경<br>우<br>나) 위‧아래턱(상ㆍ하악)의 가운데 앞니(중절치)간 최<br>대 개구(입을 벌림)운동이 '
 '1cm이하로 제한되는<br>경우<br>다) 위‧아래턱(상ㆍ하악)의 부정교합(전방, 측방)이<br>1.5cm이상인 경우<br>라) 1개 '
 '이하의 치아만 교합되는 상태<br>마) 연하기능검사(비디오 투시검사)상 연하장애가 있<br>고, 유동식 섭취 시 흡인이 발생하고 연식 '
 "외에<br>는 섭취가 불가능한 상태</p><br><p id='42' data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000954',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
