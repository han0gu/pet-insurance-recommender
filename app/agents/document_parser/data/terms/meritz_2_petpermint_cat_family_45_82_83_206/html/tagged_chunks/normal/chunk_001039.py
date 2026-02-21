from langchain_core.documents import Document

chunk = Document(
    page_content=('injury) 소견<br>이 있으면서 도수근력검사(MMT)에서 근력이 “1<br>등급(Trace)"인 경우</p><br><p '
 "id='61' data-category='paragraph' style='font-size:20px'>9) “관절 하나의 기능에 뚜렷한 "
 "장해를 남긴 때”라 함<br>은 아래의 경우 중 하나에 해당하는 때를 말한다.</p><br><p id='62' "
 "data-category='list' style='font-size:16px'>가) 해당 관절의 운동범위 합계가 "
 '정상운동범위의<br>1/2 이하로 제한된'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_001039',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
