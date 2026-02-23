from langchain_core.documents import Document

chunk = Document(
    page_content=("시력상태</p><br><p id='9' data-category='list' style='font-size:20px'>5) 안구(눈동자) "
 '운동장해의 판정은 질병의 진단 또는<br>외상 후 1년 이상이 지난 뒤에 그 장해정도를 평가한다.<br>6) “안구(눈동자)의 뚜렷한 '
 "운동장해”라 함은 아래의<br>두 경우 중 하나에 해당하는 경우를 말한다.</p><br><p id='10' "
 "data-category='list' style='font-size:20px'>가) 한 눈의 안구(눈동자)의 주시야(머리를 "
 '움직이<br>지 않고 눈만을'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'eye', 'head']},
 'indexing': {'chunk_id': 'chunk_000924',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
